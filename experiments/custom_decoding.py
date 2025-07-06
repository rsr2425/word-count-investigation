"""Custom logits processors and pipelines for word count control during generation."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import (
    LogitsProcessor, 
    LogitsProcessorList,
    Pipeline,
    PreTrainedModel, 
    PreTrainedTokenizer
)


class GracefulWordCountLogitsProcessor(LogitsProcessor):
    """A logits processor that controls word count during generation.
    
    This processor monitors the word count during generation and:
    1. Boosts punctuation tokens when approaching the target word count
    2. Forces EOS token when the target is reached or exceeded
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        target_word_count: int,
        word_count_fn: callable,
        buffer_window: int = 5,
        completion_boost: float = 5.0
    ):
        """Initialize the word count logits processor.
        
        Args:
            tokenizer: The tokenizer to use for token conversion
            target_word_count: The target number of words
            word_count_fn: Function to count words from input_ids
            buffer_window: Number of words before target to start boosting punctuation
            completion_boost: Amount to boost completion token probabilities
        """
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.target_word_count = target_word_count
        self.word_count_fn = word_count_fn
        self.buffer_window = buffer_window
        self.completion_boost = completion_boost
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply word count control to the logits.
        
        Args:
            input_ids: The current sequence of input IDs
            scores: The logits for the next token
            
        Returns:
            Modified logits with word count control applied
        """
        current_word_count = self.word_count_fn(input_ids)
        
        # If within the buffer window, boost punctuation tokens
        if self.target_word_count - self.buffer_window <= current_word_count < self.target_word_count:
            punctuation_tokens = [".", "!", "?"]
            punctuation_ids = [
                self.tokenizer.convert_tokens_to_ids(tok) 
                for tok in punctuation_tokens 
                if tok in self.tokenizer.vocab
            ]
            
            for token_id in punctuation_ids:
                scores[:, token_id] += self.completion_boost
        
        # Prevent overshooting: strongly favor EOS if count exceeds target
        if current_word_count >= self.target_word_count:
            scores[:, :] = -float("inf")  # Set all probabilities to zero
            scores[:, self.eos_token_id] = 0.0  # Make EOS the only valid option
        
        return scores


class CustomLogitsProcessorPipeline(Pipeline):
    """A custom text generation pipeline that supports custom logits processors.
    
    This pipeline allows you to inject custom logits processors into the generation
    process, enabling fine-grained control over the generation behavior.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        logits_processor: Optional[LogitsProcessorList] = None,
        **kwargs
    ):
        """Initialize the custom pipeline.
        
        Args:
            model: The pre-trained model to use
            tokenizer: The tokenizer associated with the model
            logits_processor: A custom list of logits processors to apply
            **kwargs: Additional arguments passed to the parent Pipeline class
        """
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.custom_logits_processor = logits_processor or LogitsProcessorList()
    
    def _sanitize_parameters(
        self, *args, **kwargs
    ) -> Tuple[Dict, Dict, Dict]:
        """Sanitize parameters for the pipeline methods."""
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}
        
        # Extract generation parameters
        generation_kwargs = ["max_length", "max_new_tokens", "top_k", "temperature", "do_sample"]
        for param in generation_kwargs:
            if param in kwargs:
                forward_params[param] = kwargs[param]
        
        # Extract preprocessing parameters
        preprocess_params.update(kwargs.get("preprocess_kwargs", {}))
        postprocess_params.update(kwargs.get("postprocess_kwargs", {}))
        
        return preprocess_params, forward_params, postprocess_params
    
    def preprocess(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Prepare inputs for the forward method.
        
        Args:
            input_text: The raw input text to preprocess
            **kwargs: Additional preprocessing arguments
            
        Returns:
            Preprocessed inputs in dictionary format
        """
        input_ids = self.tokenizer(
            input_text,
            return_tensors="pt",
        ).input_ids
        input_ids = input_ids.to(self.model.device)
        return {"input_ids": input_ids}
    
    def _forward(self, inputs: Dict, **generate_kwargs) -> Any:
        """Forward pass through the pipeline with custom logits processing.
        
        Args:
            inputs: The inputs to the model
            **generate_kwargs: Additional arguments passed to the generate method
            
        Returns:
            The output of the pipeline with custom logits processing applied
        """
        generate_kwargs["logits_processor"] = self.custom_logits_processor
        return self.model.generate(
            inputs["input_ids"],
            **generate_kwargs,
        )
    
    def postprocess(self, model_outputs: Any, **kwargs) -> List[str]:
        """Postprocess the model outputs.
        
        Args:
            model_outputs: The raw outputs from the model
            **kwargs: Additional arguments for postprocessing
            
        Returns:
            Postprocessed outputs as a list of strings
        """
        generated_text = self.tokenizer.batch_decode(
            model_outputs,
            skip_special_tokens=True,
        )
        return generated_text


def create_word_count_processor(
    tokenizer: PreTrainedTokenizer,
    target_word_count: int,
    assistant_only: bool = True,
    buffer_window: int = 5,
    completion_boost: float = 5.0
) -> LogitsProcessorList:
    """Create a word count logits processor for text generation.
    
    Args:
        tokenizer: The tokenizer to use
        target_word_count: The target number of words
        assistant_only: Whether to count only assistant response words
        buffer_window: Number of words before target to start boosting punctuation
        completion_boost: Amount to boost completion token probabilities
        
    Returns:
        A LogitsProcessorList containing the word count processor
    """
    def word_count_fn(input_ids):
        """Count words in the input sequence."""
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        if assistant_only:
            # Extract only the assistant's response
            assistant_marker = '<|start_header_id|>assistant<|end_header_id|>'
            if assistant_marker in decoded_text:
                decoded_text = decoded_text.split(assistant_marker)[-1]
        
        return len(decoded_text.split())
    
    processor = GracefulWordCountLogitsProcessor(
        tokenizer=tokenizer,
        target_word_count=target_word_count,
        word_count_fn=word_count_fn,
        buffer_window=buffer_window,
        completion_boost=completion_boost
    )
    
    return LogitsProcessorList([processor])


def create_custom_generation_pipeline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    target_word_count: int,
    assistant_only: bool = True,
    **generation_kwargs
) -> CustomLogitsProcessorPipeline:
    """Create a text generation pipeline with word count control.
    
    Args:
        model: The pre-trained model to use
        tokenizer: The tokenizer associated with the model
        target_word_count: The target number of words for generation
        assistant_only: Whether to count only assistant response words
        **generation_kwargs: Additional generation parameters
        
    Returns:
        A custom pipeline with word count control
    """
    logits_processor = create_word_count_processor(
        tokenizer=tokenizer,
        target_word_count=target_word_count,
        assistant_only=assistant_only
    )
    
    # Set default generation parameters
    default_kwargs = {
        "max_new_tokens": 10000,  # High limit to allow logits processor to control length
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id
    }
    default_kwargs.update(generation_kwargs)
    
    return CustomLogitsProcessorPipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        logits_processor=logits_processor,
        **default_kwargs
    )