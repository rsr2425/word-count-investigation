"""Tests for core experiment functions."""

import pytest
from unittest.mock import Mock, patch
from datasets import Dataset

from experiments.experiments import process_dataset, run_experiment
from experiments.metrics import Metric


class TestProcessDataset:
    """Test the process_dataset function."""
    
    def test_process_dataset_basic(self, sample_dataset, mock_llm):
        """Test basic dataset processing."""
        metrics = [Metric.ROUGE]
        
        result = process_dataset(
            dataset=sample_dataset,
            llm=mock_llm,
            n=5,
            metrics=metrics,
            word_count_target=None,
            otherinstructions=None
        )
        
        # Check that the dataset was processed
        assert len(result) == len(sample_dataset)
        assert "ai_summary" in result.column_names
        assert "ai_summary_word_count" in result.column_names
        assert "rouge1" in result.column_names
    
    def test_process_dataset_with_word_count(self, sample_dataset, mock_llm):
        """Test dataset processing with word count target."""
        metrics = [Metric.ROUGE]
        
        result = process_dataset(
            dataset=sample_dataset,
            llm=mock_llm,
            n=5,
            metrics=metrics,
            word_count_target=25,
            otherinstructions=None
        )
        
        # Check that word count control was applied
        assert "total_model_calls" in result.column_names
        
        # Check that summaries were generated
        for record in result:
            assert "ai_summary" in record
            assert len(record["ai_summary"]) > 0
    
    def test_process_dataset_with_instructions(self, sample_dataset, mock_llm):
        """Test dataset processing with additional instructions."""
        metrics = [Metric.ROUGE]
        
        result = process_dataset(
            dataset=sample_dataset,
            llm=mock_llm,
            n=5,
            metrics=metrics,
            word_count_target=None,
            otherinstructions="Make it concise"
        )
        
        assert len(result) == len(sample_dataset)
        assert "ai_summary" in result.column_names


class TestRunExperiment:
    """Test the run_experiment function."""
    
    @patch('experiments.experiments.ChatOpenAI')
    def test_run_experiment_basic(self, mock_chat_openai, sample_dataset):
        """Test basic experiment run."""
        # Setup mock
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        def mock_invoke(messages):
            response = Mock()
            response.content = "Test summary."
            return response
        mock_llm.invoke = mock_invoke
        
        metrics = [Metric.ROUGE]
        
        result = run_experiment(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            dataset=sample_dataset,
            number_of_questions=5,
            metrics=metrics,
            subset_size=1
        )
        
        # Check that ChatOpenAI was called with correct parameters
        mock_chat_openai.assert_called_once_with(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Check that result has expected structure
        assert len(result) == 1  # subset_size=1
        assert "ai_summary" in result.column_names
    
    @patch('experiments.experiments.ChatOpenAI')
    def test_run_experiment_with_subset(self, mock_chat_openai, sample_dataset):
        """Test experiment with dataset subset."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        def mock_invoke(messages):
            response = Mock()
            response.content = "Test summary."
            return response
        mock_llm.invoke = mock_invoke
        
        metrics = [Metric.ROUGE]
        
        result = run_experiment(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            dataset=sample_dataset,
            number_of_questions=5,
            metrics=metrics,
            subset_size=1,
            word_count_target=25
        )
        
        assert len(result) == 1
        assert "total_model_calls" in result.column_names
    
    @patch('experiments.experiments.ChatOpenAI')
    @patch('experiments.experiments.log_dataset_to_wandb')
    def test_run_experiment_with_logging(self, mock_log_wandb, mock_chat_openai, sample_dataset):
        """Test experiment with wandb logging."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        def mock_invoke(messages):
            response = Mock()
            response.content = "Test summary."
            return response
        mock_llm.invoke = mock_invoke
        
        metrics = [Metric.ROUGE]
        
        result = run_experiment(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            dataset=sample_dataset,
            number_of_questions=5,
            metrics=metrics,
            subset_size=1,
            log_to_wandb=True,
            project_name="test-project",
            run_prefix="test_"
        )
        
        # Check that logging was called
        mock_log_wandb.assert_called_once()
        call_args = mock_log_wandb.call_args[0]
        assert call_args[1] == "test-project"  # project_name
        assert call_args[2] == "test_gpt-3.5-turbo"  # run_name
    
    @patch('experiments.experiments.ChatOpenAI')
    def test_run_experiment_no_subset(self, mock_chat_openai, sample_dataset):
        """Test experiment without subset (full dataset)."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        def mock_invoke(messages):
            response = Mock()
            response.content = "Test summary."
            return response
        mock_llm.invoke = mock_invoke
        
        metrics = [Metric.ROUGE]
        
        result = run_experiment(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            dataset=sample_dataset,
            number_of_questions=5,
            metrics=metrics,
            subset_size=None  # No subset
        )
        
        # Should process full dataset
        assert len(result) == len(sample_dataset)
    
    @patch('experiments.experiments.ChatOpenAI')
    def test_run_experiment_parameters(self, mock_chat_openai, sample_dataset):
        """Test that all parameters are properly passed through."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        def mock_invoke(messages):
            response = Mock()
            response.content = "Test summary with instructions."
            return response
        mock_llm.invoke = mock_invoke
        
        metrics = [Metric.ROUGE]
        
        result = run_experiment(
            model_name="gpt-4",
            temperature=0.5,
            dataset=sample_dataset,
            number_of_questions=10,
            metrics=metrics,
            word_count_target=50,
            subset_size=1,
            otherinstructions="Be concise",
            log_to_wandb=False
        )
        
        # Verify ChatOpenAI was called with correct model and temperature
        mock_chat_openai.assert_called_once_with(
            model_name="gpt-4",
            temperature=0.5
        )
        
        assert len(result) == 1
        assert "ai_summary" in result.column_names