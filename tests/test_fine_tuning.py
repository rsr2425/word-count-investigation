"""Tests for fine-tuning utilities."""

import pytest
import torch
from unittest.mock import Mock, patch

from experiments.fine_tuning import (
    WordCountLossTrainer,
    create_quantization_config,
    create_lora_config,
    setup_model_for_training,
    create_training_config,
    train_word_count_model,
    save_model_and_tokenizer
)


class TestWordCountLossTrainer:
    """Test the WordCountLossTrainer class."""
    
    def test_init(self):
        """Test trainer initialization."""
        trainer = WordCountLossTrainer(
            target_word_count=50,
            model=Mock(),
            tokenizer=Mock()
        )
        
        assert trainer.target_word_count == 50
    
    def test_compute_loss_basic(self):
        """Test basic loss computation."""
        trainer = WordCountLossTrainer(
            target_word_count=25,
            model=Mock(),
            tokenizer=Mock()
        )
        
        # Mock model and inputs
        model = Mock()
        outputs = Mock()
        outputs.loss = torch.tensor(1.0)
        model.return_value = outputs
        
        # Mock inputs with word count markers
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]])
        }
        
        # Mock tokenizer
        trainer.tokenizer = Mock()
        trainer.tokenizer.batch_decode.return_value = [
            "Test input [WORD_COUNT]25[END_WORD_COUNT]",
            "Generated response with multiple words here"
        ]
        
        # Mock logits
        outputs.logits = torch.randn(1, 5, 1000)
        
        loss = trainer.compute_loss(model, inputs)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_compute_loss_without_word_count_markers(self):
        """Test loss computation when no word count markers present."""
        trainer = WordCountLossTrainer(
            target_word_count=30,
            model=Mock(),
            tokenizer=Mock()
        )
        
        model = Mock()
        outputs = Mock()
        outputs.loss = torch.tensor(1.5)
        model.return_value = outputs
        
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]])
        }
        
        trainer.tokenizer = Mock()
        trainer.tokenizer.batch_decode.return_value = [
            "Input without markers",
            "Generated response"
        ]
        
        outputs.logits = torch.randn(1, 5, 1000)
        
        loss = trainer.compute_loss(model, inputs)
        
        assert isinstance(loss, torch.Tensor)
    
    def test_compute_loss_with_return_outputs(self):
        """Test loss computation with return_outputs=True."""
        trainer = WordCountLossTrainer(
            target_word_count=20,
            model=Mock(),
            tokenizer=Mock()
        )
        
        model = Mock()
        outputs = Mock()
        outputs.loss = torch.tensor(0.8)
        model.return_value = outputs
        
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]])
        }
        
        trainer.tokenizer = Mock()
        trainer.tokenizer.batch_decode.return_value = [
            "Input text [WORD_COUNT]20[END_WORD_COUNT]",
            "Short response"
        ]
        
        outputs.logits = torch.randn(1, 3, 1000)
        
        loss, returned_outputs = trainer.compute_loss(model, inputs, return_outputs=True)
        
        assert isinstance(loss, torch.Tensor)
        assert returned_outputs == outputs


class TestConfigurationFunctions:
    """Test configuration creation functions."""
    
    def test_create_quantization_config_defaults(self):
        """Test quantization config with default parameters."""
        config = create_quantization_config()
        
        # Should create a config object (mocked in conftest)
        assert config is not None
    
    def test_create_quantization_config_custom(self):
        """Test quantization config with custom parameters."""
        config = create_quantization_config(
            load_in_4bit=False,
            quant_type="fp4",
            use_double_quant=False,
            compute_dtype=torch.float32
        )
        
        assert config is not None
    
    def test_create_lora_config_defaults(self):
        """Test LoRA config with default parameters."""
        config = create_lora_config()
        
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.target_modules == "all-linear"
        assert config.task_type == "CAUSAL_LM"
    
    def test_create_lora_config_custom(self):
        """Test LoRA config with custom parameters."""
        config = create_lora_config(
            r=32,
            alpha=64,
            dropout=0.2,
            target_modules=["q_proj", "v_proj"],
            task_type="SEQ_2_SEQ_LM"
        )
        
        assert config.r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.2
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.task_type == "SEQ_2_SEQ_LM"
    
    def test_create_training_config_defaults(self):
        """Test training config with default parameters."""
        config = create_training_config(output_dir="test_output")
        
        assert config.output_dir == "test_output"
        assert config.num_train_epochs == 5
        assert config.per_device_train_batch_size == 1
        assert config.learning_rate == 2e-4
        assert config.max_seq_length == 1024
    
    def test_create_training_config_custom(self):
        """Test training config with custom parameters."""
        config = create_training_config(
            output_dir="custom_output",
            num_train_epochs=10,
            per_device_train_batch_size=2,
            learning_rate=1e-4,
            max_seq_length=512,
            warmup_steps=50
        )
        
        assert config.output_dir == "custom_output"
        assert config.num_train_epochs == 10
        assert config.per_device_train_batch_size == 2
        assert config.learning_rate == 1e-4
        assert config.max_seq_length == 512
        assert config.warmup_steps == 50


class TestModelSetup:
    """Test model setup functions."""
    
    @patch('experiments.fine_tuning.get_peft_model')
    @patch('experiments.fine_tuning.AutoTokenizer')
    @patch('experiments.fine_tuning.AutoModelForCausalLM')
    def test_setup_model_for_training_basic(self, mock_model_class, mock_tokenizer_class, mock_get_peft):
        """Test basic model setup."""
        # Setup mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        model, tokenizer = setup_model_for_training("test-model")
        
        # Check calls
        mock_model_class.from_pretrained.assert_called_once_with(
            "test-model",
            quantization_config=None,
            device_map='auto'
        )
        mock_tokenizer_class.from_pretrained.assert_called_once_with("test-model")
        
        # Check tokenizer setup
        assert tokenizer.pad_token == tokenizer.eos_token
        assert tokenizer.padding_side == "right"
        
        # Should not apply LoRA without config
        mock_get_peft.assert_not_called()
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
    
    @patch('experiments.fine_tuning.get_peft_model')
    @patch('experiments.fine_tuning.AutoTokenizer')
    @patch('experiments.fine_tuning.AutoModelForCausalLM')
    def test_setup_model_for_training_with_lora(self, mock_model_class, mock_tokenizer_class, mock_get_peft):
        """Test model setup with LoRA."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_peft_model = Mock()
        mock_lora_config = Mock()
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_get_peft.return_value = mock_peft_model
        
        model, tokenizer = setup_model_for_training(
            "test-model",
            lora_config=mock_lora_config
        )
        
        # Should apply LoRA
        mock_get_peft.assert_called_once_with(mock_model, mock_lora_config)
        assert model == mock_peft_model
    
    @patch('experiments.fine_tuning.AutoTokenizer')
    @patch('experiments.fine_tuning.AutoModelForCausalLM')
    def test_setup_model_for_training_with_quantization(self, mock_model_class, mock_tokenizer_class):
        """Test model setup with quantization."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_quant_config = Mock()
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        model, tokenizer = setup_model_for_training(
            "test-model",
            quantization_config=mock_quant_config
        )
        
        # Check quantization config was passed
        mock_model_class.from_pretrained.assert_called_once_with(
            "test-model",
            quantization_config=mock_quant_config,
            device_map='auto'
        )


class TestTrainingFunctions:
    """Test training and saving functions."""
    
    def test_train_word_count_model(self):
        """Test word count model training setup."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_train_dataset = Mock()
        mock_eval_dataset = Mock()
        mock_config = Mock()
        
        def mock_formatting_func(sample):
            return f"Format: {sample}"
        
        trainer = train_word_count_model(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            formatting_func=mock_formatting_func,
            training_config=mock_config,
            target_word_count=40
        )
        
        assert isinstance(trainer, WordCountLossTrainer)
        assert trainer.target_word_count == 40
    
    def test_save_model_and_tokenizer_basic(self):
        """Test basic model and tokenizer saving."""
        mock_trainer = Mock()
        mock_trainer.model = Mock()
        mock_trainer.tokenizer = Mock()
        
        # Mock that model doesn't have merge_and_unload
        del mock_trainer.model.merge_and_unload
        
        save_model_and_tokenizer(
            trainer=mock_trainer,
            hub_model_id="test/model",
            merge_and_unload=True
        )
        
        # Check basic saves
        mock_trainer.push_to_hub.assert_called_once_with("test/model")
        mock_trainer.tokenizer.push_to_hub.assert_called_once_with("test/model")
    
    def test_save_model_and_tokenizer_with_merge(self):
        """Test model saving with merge and unload."""
        mock_trainer = Mock()
        mock_merged_model = Mock()
        mock_trainer.model.merge_and_unload.return_value = mock_merged_model
        
        save_model_and_tokenizer(
            trainer=mock_trainer,
            hub_model_id="test/model",
            merge_and_unload=True
        )
        
        # Check merge and save
        mock_trainer.model.merge_and_unload.assert_called_once()
        mock_merged_model.push_to_hub.assert_called_once_with(
            "test/model", 
            safe_serialization=True
        )
    
    def test_save_model_and_tokenizer_no_merge(self):
        """Test model saving without merge."""
        mock_trainer = Mock()
        
        save_model_and_tokenizer(
            trainer=mock_trainer,
            hub_model_id="test/model",
            merge_and_unload=False
        )
        
        # Should not call merge_and_unload
        mock_trainer.model.merge_and_unload.assert_not_called()


class TestIntegration:
    """Integration tests for fine-tuning components."""
    
    def test_complete_fine_tuning_setup(self, sample_dataset):
        """Test complete fine-tuning setup flow."""
        # Create configurations
        quant_config = create_quantization_config()
        lora_config = create_lora_config(r=8, alpha=16)
        training_config = create_training_config(
            output_dir="test_output",
            num_train_epochs=1,
            per_device_train_batch_size=1
        )
        
        # All should be created without errors
        assert quant_config is not None
        assert lora_config is not None
        assert training_config is not None
        
        # Check configurations
        assert lora_config.r == 8
        assert lora_config.lora_alpha == 16
        assert training_config.num_train_epochs == 1
    
    @patch('experiments.fine_tuning.setup_model_for_training')
    def test_training_workflow(self, mock_setup_model, sample_dataset):
        """Test the complete training workflow."""
        # Mock model setup
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer)
        
        # Create configs
        lora_config = create_lora_config()
        training_config = create_training_config(output_dir="test")
        
        def mock_formatting_func(sample):
            return f"Formatted: {sample['text']}"
        
        # Train model
        trainer = train_word_count_model(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=sample_dataset,
            eval_dataset=sample_dataset,
            formatting_func=mock_formatting_func,
            training_config=training_config,
            target_word_count=30,
            peft_config=lora_config
        )
        
        assert isinstance(trainer, WordCountLossTrainer)
        assert trainer.target_word_count == 30