"""Fine-tuning utilities for word count controlled text generation."""

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


class WordCountLossTrainer(SFTTrainer):
    """A custom trainer that incorporates word count loss into the training objective.
    
    This trainer extends SFTTrainer to add a loss component that penalizes
    deviations from target word counts specified in the training data.
    """
    
    def __init__(self, target_word_count: int = 50, *args, **kwargs):
        """Initialize the WordCountLossTrainer.
        
        Args:
            target_word_count: The default target word count for training
            *args: Arguments passed to parent SFTTrainer
            **kwargs: Keyword arguments passed to parent SFTTrainer
        """
        super().__init__(*args, **kwargs)
        self.target_word_count = target_word_count
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with word count penalty.
        
        Args:
            model: The model being trained
            inputs: The input batch
            return_outputs: Whether to return model outputs
            **kwargs: Additional keyword arguments
            
        Returns:
            The computed loss (and outputs if return_outputs=True)
        """
        # Forward pass
        outputs = model(**inputs)
        
        # Extract input IDs and logits
        input_ids = inputs["input_ids"]
        logits = outputs.logits
        
        # Decode input prompts and responses
        tokenizer = getattr(self.data_collator, 'tokenizer', self.tokenizer)
        decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Decode generated responses
        predicted_tokens = torch.argmax(logits, dim=-1)
        decoded_responses = tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
        
        # Extract word count targets from inputs
        word_count_targets = []
        for decoded_input in decoded_inputs:
            if "[WORD_COUNT]" in decoded_input and "[END_WORD_COUNT]" in decoded_input:
                start = decoded_input.find("[WORD_COUNT]") + len("[WORD_COUNT]")
                end = decoded_input.find("[END_WORD_COUNT]")
                try:
                    word_count = int(decoded_input[start:end].strip())
                    word_count_targets.append(word_count)
                except (ValueError, IndexError) as e:
                    print(f"Error parsing word count from input: {decoded_input[:100]}...")
                    print(f"Error details: {str(e)}")
                    word_count_targets.append(self.target_word_count)
            else:
                word_count_targets.append(self.target_word_count)
        
        # Compute actual word counts from responses
        word_count_actuals = [len(response.split()) for response in decoded_responses]
        
        # Convert to tensors and ensure gradient tracking
        word_count_targets = torch.tensor(
            word_count_targets, 
            device=outputs.loss.device, 
            dtype=torch.float,
            requires_grad=False
        )
        word_count_actuals = torch.tensor(
            word_count_actuals, 
            device=outputs.loss.device, 
            dtype=torch.float,
            requires_grad=True
        )
        
        # Calculate word count loss (mean absolute error)
        word_count_loss = torch.mean(torch.abs(word_count_targets - word_count_actuals))
        
        # Combine with original loss (if available)
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            total_loss = outputs.loss + word_count_loss
        else:
            total_loss = word_count_loss
        
        return (total_loss, outputs) if return_outputs else total_loss


def create_quantization_config(
    load_in_4bit: bool = True,
    quant_type: str = "nf4",
    use_double_quant: bool = True,
    compute_dtype: torch.dtype = torch.float16
) -> BitsAndBytesConfig:
    """Create a quantization configuration for efficient training.
    
    Args:
        load_in_4bit: Whether to load model in 4-bit precision
        quant_type: Type of quantization to use
        use_double_quant: Whether to use double quantization
        compute_dtype: Compute dtype for quantized operations
        
    Returns:
        BitsAndBytesConfig object for model loading
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def create_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: str = "all-linear",
    task_type: str = "CAUSAL_LM"
) -> LoraConfig:
    """Create a LoRA configuration for parameter-efficient fine-tuning.
    
    Args:
        r: LoRA rank (higher = more parameters, better performance)
        alpha: LoRA scaling parameter (typically r * 2)
        dropout: Dropout rate for LoRA layers
        target_modules: Which modules to apply LoRA to
        task_type: Type of task for LoRA adaptation
        
    Returns:
        LoraConfig object for PEFT
    """
    return LoraConfig(
        lora_alpha=alpha,
        lora_dropout=dropout,
        r=r,
        bias="none",
        target_modules=target_modules,
        task_type=task_type
    )


def setup_model_for_training(
    model_id: str,
    lora_config: Optional[LoraConfig] = None,
    quantization_config: Optional[BitsAndBytesConfig] = None
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Set up a model and tokenizer for word count training.
    
    Args:
        model_id: HuggingFace model identifier
        lora_config: LoRA configuration for PEFT (optional)
        quantization_config: Quantization configuration (optional)
        
    Returns:
        Tuple of (model, tokenizer) ready for training
    """
    # Load model with optional quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map='auto',
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Apply LoRA if specified
    if lora_config is not None:
        model = get_peft_model(model, lora_config)
    
    return model, tokenizer


def create_training_config(
    output_dir: str,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 1,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
    warmup_steps: int = 30,
    logging_steps: int = 10,
    eval_strategy: str = "epoch",
    eval_steps: Optional[int] = None,
    lr_scheduler_type: str = 'cosine',
    **kwargs
) -> SFTConfig:
    """Create a training configuration for SFT.
    
    Args:
        output_dir: Directory to save model outputs
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        learning_rate: Learning rate for optimization
        max_seq_length: Maximum sequence length
        warmup_steps: Number of warmup steps
        logging_steps: Steps between logging
        eval_strategy: Evaluation strategy ("epoch" or "steps")
        eval_steps: Steps between evaluations (if eval_strategy="steps")
        lr_scheduler_type: Type of learning rate scheduler
        **kwargs: Additional training arguments
        
    Returns:
        SFTConfig object for training
    """
    config_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "warmup_steps": warmup_steps,
        "logging_steps": logging_steps,
        "eval_strategy": eval_strategy,
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "dataset_kwargs": {
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        "max_seq_length": max_seq_length,
        "packing": True,
    }
    
    if eval_steps is not None:
        config_kwargs["eval_steps"] = eval_steps
    
    config_kwargs.update(kwargs)
    
    return SFTConfig(**config_kwargs)


def train_word_count_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset,
    eval_dataset,
    formatting_func: callable,
    training_config: SFTConfig,
    target_word_count: int = 50,
    peft_config: Optional[LoraConfig] = None
) -> WordCountLossTrainer:
    """Train a model for word count control.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer to use
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        formatting_func: Function to format training examples
        training_config: Training configuration
        target_word_count: Default target word count
        peft_config: PEFT configuration (optional)
        
    Returns:
        Trained WordCountLossTrainer instance
    """
    trainer = WordCountLossTrainer(
        target_word_count=target_word_count,
        model=model,
        peft_config=peft_config,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    return trainer


def save_model_and_tokenizer(
    trainer: WordCountLossTrainer,
    hub_model_id: str,
    merge_and_unload: bool = True
) -> None:
    """Save the trained model and tokenizer to HuggingFace Hub.
    
    Args:
        trainer: The trained model trainer
        hub_model_id: HuggingFace Hub model ID (e.g., "username/model-name")
        merge_and_unload: Whether to merge LoRA weights before saving
    """
    # Push trainer to hub
    trainer.push_to_hub(hub_model_id)
    
    # If using LoRA, merge and push the full model
    if merge_and_unload and hasattr(trainer.model, 'merge_and_unload'):
        merged_model = trainer.model.merge_and_unload()
        merged_model.push_to_hub(hub_model_id, safe_serialization=True)
    
    # Push tokenizer
    trainer.tokenizer.push_to_hub(hub_model_id)