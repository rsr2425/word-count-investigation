"""Tests for custom decoding functionality."""

import pytest
import torch
from unittest.mock import Mock, patch

from experiments.custom_decoding import (
    GracefulWordCountLogitsProcessor,
    CustomLogitsProcessorPipeline,
    create_word_count_processor,
    create_custom_generation_pipeline
)


class TestGracefulWordCountLogitsProcessor:
    """Test the GracefulWordCountLogitsProcessor class."""
    
    def test_init(self, mock_tokenizer):
        """Test processor initialization."""
        def word_count_fn(input_ids):
            return 10
        
        processor = GracefulWordCountLogitsProcessor(
            tokenizer=mock_tokenizer,
            target_word_count=25,
            word_count_fn=word_count_fn,
            buffer_window=5,
            completion_boost=2.0
        )
        
        assert processor.target_word_count == 25
        assert processor.buffer_window == 5
        assert processor.completion_boost == 2.0
        assert processor.eos_token_id == mock_tokenizer.eos_token_id
    
    def test_call_under_target(self, mock_tokenizer):
        """Test processor when word count is under target."""
        def word_count_fn(input_ids):
            return 10  # Well under target of 25
        
        processor = GracefulWordCountLogitsProcessor(
            tokenizer=mock_tokenizer,
            target_word_count=25,
            word_count_fn=word_count_fn
        )
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.zeros(1, 1000)
        
        result = processor(input_ids, scores)
        
        # Should not modify scores significantly when well under target
        assert torch.allclose(result, scores, atol=1e-6)
    
    def test_call_in_buffer_window(self, mock_tokenizer):
        """Test processor when word count is in buffer window."""
        def word_count_fn(input_ids):
            return 22  # Within buffer window (25-5=20 to 25)
        
        processor = GracefulWordCountLogitsProcessor(
            tokenizer=mock_tokenizer,
            target_word_count=25,
            word_count_fn=word_count_fn,
            completion_boost=5.0
        )
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.zeros(1, 1000)
        
        result = processor(input_ids, scores)
        
        # Should boost punctuation tokens
        punctuation_ids = [13, 0, 30]  # Based on mock tokenizer vocab
        for token_id in punctuation_ids:
            if token_id != -1:  # Valid token
                assert result[0, token_id] > scores[0, token_id]
    
    def test_call_over_target(self, mock_tokenizer):
        """Test processor when word count exceeds target."""
        def word_count_fn(input_ids):
            return 30  # Over target of 25
        
        processor = GracefulWordCountLogitsProcessor(
            tokenizer=mock_tokenizer,
            target_word_count=25,
            word_count_fn=word_count_fn
        )
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.zeros(1, 1000)
        
        result = processor(input_ids, scores)
        
        # Should force EOS token
        eos_id = mock_tokenizer.eos_token_id
        assert result[0, eos_id] == 0.0
        
        # All other tokens should be -inf
        non_eos_mask = torch.arange(1000) != eos_id
        assert torch.all(result[0, non_eos_mask] == float('-inf'))
    
    def test_call_exactly_at_target(self, mock_tokenizer):
        """Test processor when word count exactly meets target."""
        def word_count_fn(input_ids):
            return 25  # Exactly at target
        
        processor = GracefulWordCountLogitsProcessor(
            tokenizer=mock_tokenizer,
            target_word_count=25,
            word_count_fn=word_count_fn
        )
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.zeros(1, 1000)
        
        result = processor(input_ids, scores)
        
        # Should force EOS when exactly at target
        eos_id = mock_tokenizer.eos_token_id
        assert result[0, eos_id] == 0.0


class TestCustomLogitsProcessorPipeline:
    """Test the CustomLogitsProcessorPipeline class."""
    
    def test_init(self, mock_model, mock_tokenizer):
        """Test pipeline initialization."""
        pipeline = CustomLogitsProcessorPipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            task="text-generation"
        )
        
        assert pipeline.model == mock_model
        assert pipeline.tokenizer == mock_tokenizer
    
    def test_sanitize_parameters(self, mock_model, mock_tokenizer):
        """Test parameter sanitization."""
        pipeline = CustomLogitsProcessorPipeline(
            model=mock_model,
            tokenizer=mock_tokenizer
        )
        
        _, forward, _ = pipeline._sanitize_parameters(
            max_length=100,
            temperature=0.7,
            top_k=50
        )
        
        assert forward["max_length"] == 100
        assert forward["temperature"] == 0.7
        assert forward["top_k"] == 50
    
    def test_preprocess(self, mock_model, mock_tokenizer):
        """Test preprocessing step."""
        pipeline = CustomLogitsProcessorPipeline(
            model=mock_model,
            tokenizer=mock_tokenizer
        )
        
        result = pipeline.preprocess("Test input text")
        
        assert "input_ids" in result
        assert isinstance(result["input_ids"], torch.Tensor)
    
    def test_forward(self, mock_model, mock_tokenizer):
        """Test forward pass."""
        pipeline = CustomLogitsProcessorPipeline(
            model=mock_model,
            tokenizer=mock_tokenizer
        )
        
        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        result = pipeline._forward(inputs, max_length=50)
        
        # Should call model.generate
        assert isinstance(result, torch.Tensor)
    
    def test_postprocess(self, mock_model, mock_tokenizer):
        """Test postprocessing step."""
        pipeline = CustomLogitsProcessorPipeline(
            model=mock_model,
            tokenizer=mock_tokenizer
        )
        
        model_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        result = pipeline.postprocess(model_outputs)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], str)


class TestUtilityFunctions:
    """Test utility functions for custom decoding."""
    
    def test_create_word_count_processor(self, mock_tokenizer):
        """Test word count processor creation."""
        processor_list = create_word_count_processor(
            tokenizer=mock_tokenizer,
            target_word_count=30,
            assistant_only=True
        )
        
        assert len(processor_list) == 1
        processor = processor_list[0]
        assert isinstance(processor, GracefulWordCountLogitsProcessor)
        assert processor.target_word_count == 30
    
    def test_create_word_count_processor_assistant_only(self, mock_tokenizer):
        """Test word count processor with assistant_only=True."""
        processor_list = create_word_count_processor(
            tokenizer=mock_tokenizer,
            target_word_count=25,
            assistant_only=True
        )
        
        processor = processor_list[0]
        
        # Test the word count function
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        word_count = processor.word_count_fn(input_ids)
        assert isinstance(word_count, int)
        assert word_count >= 0
    
    def test_create_word_count_processor_full_text(self, mock_tokenizer):
        """Test word count processor with assistant_only=False."""
        processor_list = create_word_count_processor(
            tokenizer=mock_tokenizer,
            target_word_count=25,
            assistant_only=False
        )
        
        processor = processor_list[0]
        
        # Test the word count function
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        word_count = processor.word_count_fn(input_ids)
        assert isinstance(word_count, int)
        assert word_count >= 0
    
    @patch('experiments.custom_decoding.create_word_count_processor')
    def test_create_custom_generation_pipeline(self, mock_create_processor, mock_model, mock_tokenizer):
        """Test custom generation pipeline creation."""
        # Mock the processor creation
        mock_processor_list = Mock()
        mock_create_processor.return_value = mock_processor_list
        
        pipeline = create_custom_generation_pipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            target_word_count=40,
            assistant_only=True,
            max_new_tokens=500,
            do_sample=True
        )
        
        # Check that processor was created with correct parameters
        mock_create_processor.assert_called_once_with(
            tokenizer=mock_tokenizer,
            target_word_count=40,
            assistant_only=True
        )
        
        assert isinstance(pipeline, CustomLogitsProcessorPipeline)
        assert pipeline.model == mock_model
        assert pipeline.tokenizer == mock_tokenizer
    
    def test_create_custom_generation_pipeline_defaults(self, mock_model, mock_tokenizer):
        """Test custom generation pipeline with default parameters."""
        pipeline = create_custom_generation_pipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            target_word_count=25
        )
        
        assert isinstance(pipeline, CustomLogitsProcessorPipeline)
        assert hasattr(pipeline, 'custom_logits_processor')


class TestIntegration:
    """Integration tests for custom decoding components."""
    
    def test_end_to_end_word_count_control(self, mock_model, mock_tokenizer):
        """Test end-to-end word count control."""
        # Create a pipeline with word count control
        pipeline = create_custom_generation_pipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            target_word_count=20,
            assistant_only=False
        )
        
        # Test that it can process input
        input_text = "Summarize this text for me please."
        
        # The pipeline should work without errors
        try:
            # Preprocess
            inputs = pipeline.preprocess(input_text)
            assert "input_ids" in inputs
            
            # Forward pass (simplified test)
            outputs = pipeline._forward(inputs, max_length=50)
            assert isinstance(outputs, torch.Tensor)
            
            # Postprocess
            result = pipeline.postprocess(outputs)
            assert isinstance(result, list)
            assert len(result) > 0
            
        except Exception as e:
            pytest.fail(f"End-to-end test failed: {e}")
    
    def test_processor_with_different_targets(self, mock_tokenizer):
        """Test processor behavior with different word count targets."""
        targets = [10, 25, 50, 100]
        
        for target in targets:
            processor_list = create_word_count_processor(
                tokenizer=mock_tokenizer,
                target_word_count=target
            )
            
            processor = processor_list[0]
            assert processor.target_word_count == target
            
            # Test with different word counts
            for current_count in [target - 10, target - 2, target, target + 5]:
                def word_count_fn(input_ids):
                    return current_count
                
                processor.word_count_fn = word_count_fn
                
                input_ids = torch.tensor([[1, 2, 3]])
                scores = torch.zeros(1, 1000)
                
                result = processor(input_ids, scores)
                
                if current_count >= target:
                    # Should force EOS
                    eos_id = mock_tokenizer.eos_token_id
                    assert result[0, eos_id] == 0.0
                else:
                    # Should not force EOS
                    assert not torch.all(result == float('-inf'))