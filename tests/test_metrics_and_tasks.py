"""Tests for metrics and tasks modules."""

import pytest
from unittest.mock import Mock, patch

from experiments.metrics import (
    Metric,
    compute_rouge,
    generate_questions,
    generate_anwsers,
    compute_factual_consistency,
    metric_fn_mapping
)
from experiments.tasks import summarize, count_words
from experiments.word_count_control import WordCountControlRunnable, create_word_count_controlled_chain


class TestMetric:
    """Test the Metric enum."""
    
    def test_metric_enum_values(self):
        """Test that Metric enum has expected values."""
        assert Metric.ROUGE.value == "ROUGE"
        assert Metric.FACTUAL_CONSISTENCY.value == "Factual Consistency"
    
    def test_metric_str_representation(self):
        """Test string representation of metrics."""
        assert str(Metric.ROUGE) == "ROUGE"
        assert str(Metric.FACTUAL_CONSISTENCY) == "Factual Consistency"


class TestComputeRouge:
    """Test the compute_rouge function."""
    
    def test_compute_rouge(self):
        """Test ROUGE computation."""
        record = {
            "ai_summary": "This is a generated summary.",
            "summary": "This is the reference summary."
        }
        
        result = compute_rouge(record)
        
        # Should return ROUGE scores
        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result
        assert "rougeLsum" in result
        
        # Scores should be floats
        assert isinstance(result["rouge1"], float)
        assert isinstance(result["rouge2"], float)
        assert isinstance(result["rougeL"], float)
        assert isinstance(result["rougeLsum"], float)
    
    def test_compute_rouge_with_kwargs(self):
        """Test ROUGE computation with additional kwargs."""
        record = {
            "ai_summary": "Generated summary text.",
            "summary": "Reference summary text."
        }
        
        # Should work with additional kwargs
        result = compute_rouge(record, extra_param="test")
        
        assert "rouge1" in result


class TestGenerateQuestions:
    """Test the generate_questions function."""
    
    def test_generate_questions_success(self):
        """Test successful question generation."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '{"questions": ["Is this true?", "Does this happen?"]}'
        mock_llm.invoke.return_value = mock_response
        
        text = "Sample text for question generation."
        n = 2
        
        questions = generate_questions(text, mock_llm, n)
        
        assert len(questions) == 2
        assert "Is this true?" in questions
        assert "Does this happen?" in questions
        
        # Check that LLM was called correctly
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0][0] == "system"
        assert call_args[1][0] == "human"
        assert text in call_args[1][1]
    
    def test_generate_questions_json_error(self):
        """Test question generation with JSON decode error."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = 'Invalid JSON'
        mock_llm.invoke.return_value = mock_response
        
        text = "Sample text."
        n = 3
        
        questions = generate_questions(text, mock_llm, n)
        
        # Should return a dict with empty questions on JSON error
        assert isinstance(questions, dict)
        assert "questions" in questions


class TestGenerateAnswers:
    """Test the generate_anwsers function."""
    
    def test_generate_answers_success(self):
        """Test successful answer generation."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '{"answers": ["Yes", "No", "idk"]}'
        mock_llm.invoke.return_value = mock_response
        
        questions = ["Question 1?", "Question 2?", "Question 3?"]
        source_text = "Source text for answering questions."
        
        answers = generate_anwsers(questions, source_text, mock_llm)
        
        assert len(answers) == 3
        assert "Yes" in answers
        assert "No" in answers
        assert "idk" in answers
    
    def test_generate_answers_json_error(self):
        """Test answer generation with JSON decode error."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = 'Invalid JSON'
        mock_llm.invoke.return_value = mock_response
        
        questions = ["Question 1?", "Question 2?"]
        source_text = "Source text."
        
        answers = generate_anwsers(questions, source_text, mock_llm)
        
        # Should return list of 'idk' on JSON error
        assert len(answers) == 2
        assert all(answer == "idk" for answer in answers)
    
    def test_generate_answers_type_error(self):
        """Test answer generation with type error."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '{"answers": null}'  # Will cause TypeError
        mock_llm.invoke.return_value = mock_response
        
        questions = ["Question?"]
        source_text = "Source text."
        
        answers = generate_anwsers(questions, source_text, mock_llm)
        
        assert len(answers) == 1
        assert answers[0] == "idk"


class TestComputeFactualConsistency:
    """Test the compute_factual_consistency function."""
    
    def test_compute_factual_consistency(self):
        """Test factual consistency computation."""
        mock_llm = Mock()
        
        # Mock responses for different calls
        responses = [
            '{"questions": ["Is A true?", "Is B correct?"]}',  # generate_questions
            '{"answers": ["Yes", "No"]}',  # gt_answers
            '{"answers": ["Yes", "idk"]}',  # human_summary_answers
            '{"answers": ["No", "No"]}'   # ai_summary_answers
        ]
        
        mock_responses = [Mock() for _ in responses]
        for i, content in enumerate(responses):
            mock_responses[i].content = content
        
        mock_llm.invoke.side_effect = mock_responses
        
        record = {
            "text": "Source text",
            "summary": "Human summary",
            "ai_summary": "AI summary"
        }
        
        result = compute_factual_consistency(record, mock_llm, 2)
        
        # Check return structure
        assert "gt_answers" in result
        assert "human_summary_answers" in result
        assert "ai_summary_answers" in result
        assert "human_factual_consistency" in result
        assert "ai_factual_consistency" in result
        
        # Check consistency scores
        assert isinstance(result["human_factual_consistency"], float)
        assert isinstance(result["ai_factual_consistency"], float)
        assert 0 <= result["human_factual_consistency"] <= 1
        assert 0 <= result["ai_factual_consistency"] <= 1
    
    def test_compute_factual_consistency_all_idk_human(self):
        """Test factual consistency when human answers are all 'idk'."""
        mock_llm = Mock()
        
        responses = [
            '{"questions": ["Question?"]}',
            '{"answers": ["Yes"]}',  # gt_answers
            '{"answers": ["idk"]}',  # human_summary_answers (all idk)
            '{"answers": ["Yes"]}'   # ai_summary_answers
        ]
        
        mock_responses = [Mock() for _ in responses]
        for i, content in enumerate(responses):
            mock_responses[i].content = content
        
        mock_llm.invoke.side_effect = mock_responses
        
        record = {
            "text": "Source text",
            "summary": "Human summary",
            "ai_summary": "AI summary"
        }
        
        result = compute_factual_consistency(record, mock_llm, 1)
        
        # Human consistency should be 0 when all answers are 'idk'
        assert result["human_factual_consistency"] == 0
    
    def test_compute_factual_consistency_all_idk_ai(self):
        """Test factual consistency when AI answers are all 'idk'."""
        mock_llm = Mock()
        
        responses = [
            '{"questions": ["Question?"]}',
            '{"answers": ["Yes"]}',  # gt_answers
            '{"answers": ["Yes"]}',  # human_summary_answers
            '{"answers": ["idk"]}'   # ai_summary_answers (all idk)
        ]
        
        mock_responses = [Mock() for _ in responses]
        for i, content in enumerate(responses):
            mock_responses[i].content = content
        
        mock_llm.invoke.side_effect = mock_responses
        
        record = {
            "text": "Source text",
            "summary": "Human summary", 
            "ai_summary": "AI summary"
        }
        
        result = compute_factual_consistency(record, mock_llm, 1)
        
        # AI consistency should be 0 when all answers are 'idk'
        assert result["ai_factual_consistency"] == 0


class TestMetricFnMapping:
    """Test the metric_fn_mapping."""
    
    def test_metric_fn_mapping_contains_all_metrics(self):
        """Test that mapping contains all metric types."""
        assert Metric.ROUGE in metric_fn_mapping
        assert Metric.FACTUAL_CONSISTENCY in metric_fn_mapping
    
    def test_metric_fn_mapping_functions(self):
        """Test that mapping contains callable functions."""
        assert callable(metric_fn_mapping[Metric.ROUGE])
        assert callable(metric_fn_mapping[Metric.FACTUAL_CONSISTENCY])
        
        # Test that they are the expected functions
        assert metric_fn_mapping[Metric.ROUGE] == compute_rouge
        assert metric_fn_mapping[Metric.FACTUAL_CONSISTENCY] == compute_factual_consistency


class TestCountWords:
    """Test the count_words function."""
    
    def test_count_words_basic(self):
        """Test basic word counting."""
        text = "This is a test sentence."
        count = count_words(text)
        assert count == 5
    
    def test_count_words_empty(self):
        """Test word counting with empty string."""
        assert count_words("") == 0  # Empty string has no words
    
    def test_count_words_whitespace(self):
        """Test word counting with whitespace."""
        assert count_words("   ") == 0  # Whitespace has no words
        assert count_words("word1 word2") == 2
        assert count_words("word1    word2") == 2  # Multiple spaces
    
    def test_count_words_single_word(self):
        """Test word counting with single word."""
        assert count_words("word") == 1
    
    def test_count_words_with_punctuation(self):
        """Test word counting with punctuation."""
        text = "Hello, world! How are you?"
        count = count_words(text)
        assert count == 5  # Punctuation attached to words


class TestSummarize:
    """Test the summarize function."""
    
    def test_summarize_without_word_count_target(self):
        """Test summarization without word count target."""
        record = {
            "text": "This is a long article that needs to be summarized effectively.",
            "summary": "Article summary."
        }
        
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Generated summary of the article."
        mock_llm.invoke.return_value = mock_response
        
        result = summarize(record, mock_llm)
        
        # Check return structure
        assert "text_word_count" in result
        assert "summary_word_count" in result
        assert "ai_summary" in result
        assert "ai_summary_word_count" in result
        assert "total_model_calls" in result
        
        # Check values
        assert result["text_word_count"] == 11
        assert result["summary_word_count"] == 2
        assert result["ai_summary"] == "Generated summary of the article."
        assert result["ai_summary_word_count"] == 5
        assert result["total_model_calls"] == 1
    
    def test_summarize_with_word_count_target(self):
        """Test summarization with word count target."""
        record = {
            "text": "Article text for summarization.",
            "summary": "Summary."
        }
        
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Controlled summary."
        mock_llm.invoke.return_value = mock_response
        
        # Mock the word count control chain
        with patch('experiments.tasks.create_word_count_controlled_chain') as mock_create_chain:
            mock_chain = Mock()
            mock_chain.invoke.return_value = {
                "final_summary": "Word count controlled summary.",
                "attempts": 3
            }
            mock_create_chain.return_value = mock_chain
            
            result = summarize(record, mock_llm, word_count_target=25)
            
            # Check that word count control was used
            mock_create_chain.assert_called_once_with(
                llm=mock_llm,
                word_count_target=25,
                tolerance=10,
                max_revision_attempts=5
            )
            
            assert result["ai_summary"] == "Word count controlled summary."
            assert result["total_model_calls"] == 3
    
    def test_summarize_with_other_instructions(self):
        """Test summarization with additional instructions."""
        record = {
            "text": "Text to summarize",
            "summary": "Summary"
        }
        
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Summary with instructions."
        mock_llm.invoke.return_value = mock_response
        
        result = summarize(record, mock_llm, otherinstructions="Be concise")
        
        # Check that LLM was called
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        
        # Check that instructions were included in system message
        system_message = call_args[0][1]
        assert "Be concise" in system_message
    
    def test_summarize_with_all_parameters(self):
        """Test summarization with all parameters."""
        record = {
            "text": "Test article text",
            "summary": "Test summary"
        }
        
        mock_llm = Mock()
        
        with patch('experiments.tasks.create_word_count_controlled_chain') as mock_create_chain:
            mock_chain = Mock()
            mock_chain.invoke.return_value = {
                "final_summary": "Complete summary.",
                "attempts": 2
            }
            mock_create_chain.return_value = mock_chain
            
            result = summarize(
                record=record,
                llm=mock_llm,
                word_count_target=30,
                otherinstructions="Be precise",
                tolerance=5,
                max_revision_attempts=3
            )
            
            # Check that all parameters were passed
            mock_create_chain.assert_called_once_with(
                llm=mock_llm,
                word_count_target=30,
                tolerance=5,
                max_revision_attempts=3
            )


class TestWordCountControlRunnable:
    """Test the WordCountControlRunnable class."""
    
    def test_init(self):
        """Test WordCountControlRunnable initialization."""
        mock_llm = Mock()
        
        runnable = WordCountControlRunnable(
            llm=mock_llm,
            word_count_target=30,
            tolerance=5,
            max_revision_attempts=3
        )
        
        assert runnable.llm == mock_llm
        assert runnable.word_count_target == 30
        assert runnable.tolerance == 5
        assert runnable.max_revision_attempts == 3
    
    def test_count_words_method(self):
        """Test the _count_words method."""
        mock_llm = Mock()
        runnable = WordCountControlRunnable(llm=mock_llm)
        
        assert runnable._count_words("hello world") == 2
        assert runnable._count_words("") == 0
    
    def test_is_within_tolerance(self):
        """Test the _is_within_tolerance method."""
        mock_llm = Mock()
        runnable = WordCountControlRunnable(
            llm=mock_llm,
            word_count_target=20,
            tolerance=5
        )
        
        # Test with 15 words (within tolerance: 20 ± 5 = 15-25)
        fifteen_word_text = " ".join(["word"] * 15)
        assert runnable._is_within_tolerance(fifteen_word_text)  # 15 words, within tolerance
        
        # Test with 20 words (exact target)
        twenty_word_text = " ".join(["word"] * 20)
        assert runnable._is_within_tolerance(twenty_word_text)  # 20 words, exact target
        
        # Test with 30 words (outside tolerance)
        thirty_word_text = " ".join(["word"] * 30)
        assert not runnable._is_within_tolerance(thirty_word_text)  # 30 words, outside tolerance
    
    def test_invoke_string_input(self):
        """Test invoke with string input."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Summary response."  # 2 words, will be outside tolerance
        mock_llm.invoke.return_value = mock_response
        
        runnable = WordCountControlRunnable(
            llm=mock_llm,
            word_count_target=10,
            tolerance=2,
            max_revision_attempts=1
        )
        
        result = runnable.invoke("Input text for summarization")
        
        assert "final_summary" in result
        assert "attempts" in result
        assert result["attempts"] == 1
    
    def test_invoke_dict_input(self):
        """Test invoke with dictionary input."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Dictionary input summary response."  # 4 words
        mock_llm.invoke.return_value = mock_response
        
        runnable = WordCountControlRunnable(llm=mock_llm, word_count_target=5, tolerance=2)
        
        input_dict = {"sample_text": "Sample text for processing"}
        result = runnable.invoke(input_dict)
        
        assert result["final_summary"] == "Dictionary input summary response."
    
    def test_invoke_with_revision(self):
        """Test invoke that requires revision."""
        mock_llm = Mock()
        
        # First response is too long, second is better
        responses = [
            Mock(content="This is a very long response that exceeds the target word count significantly"),
            Mock(content="Shorter response here")
        ]
        mock_llm.invoke.side_effect = responses
        
        runnable = WordCountControlRunnable(
            llm=mock_llm,
            word_count_target=5,
            tolerance=2,
            max_revision_attempts=2
        )
        
        result = runnable.invoke("Test input")
        
        # Should have made 2 attempts
        assert result["attempts"] == 2
        assert mock_llm.invoke.call_count == 2


class TestCreateWordCountControlledChain:
    """Test the create_word_count_controlled_chain function."""
    
    def test_create_word_count_controlled_chain(self):
        """Test creating word count controlled chain."""
        mock_llm = Mock()
        
        chain = create_word_count_controlled_chain(
            llm=mock_llm,
            word_count_target=25,
            tolerance=5,
            max_revision_attempts=3
        )
        
        # Should return a runnable chain
        assert hasattr(chain, 'invoke')
    
    @patch('experiments.word_count_control.StrOutputParser')
    def test_create_word_count_controlled_chain_structure(self, mock_parser):
        """Test that chain is created with correct structure."""
        mock_llm = Mock()
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        
        chain = create_word_count_controlled_chain(llm=mock_llm)
        
        # StrOutputParser should be instantiated
        mock_parser.assert_called_once()


class TestIntegration:
    """Integration tests for metrics and tasks."""
    
    def test_summarize_with_rouge_evaluation(self):
        """Test complete workflow: summarize + ROUGE evaluation."""
        record = {
            "text": "Sample article for testing integration.",
            "summary": "Test article."
        }
        
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Generated test summary."
        mock_llm.invoke.return_value = mock_response
        
        # Summarize
        summary_result = summarize(record, mock_llm)
        
        # Add AI summary to record for ROUGE
        record["ai_summary"] = summary_result["ai_summary"]
        
        # Compute ROUGE
        rouge_result = compute_rouge(record)
        
        # Check complete workflow
        assert summary_result["ai_summary"] == "Generated test summary."
        assert "rouge1" in rouge_result
        assert isinstance(rouge_result["rouge1"], float)
    
    def test_metric_fn_mapping_integration(self):
        """Test that metric function mapping works correctly."""
        record = {
            "text": "Test text",
            "summary": "Reference",
            "ai_summary": "Generated"
        }
        
        mock_llm = Mock()
        
        # Test ROUGE metric
        rouge_fn = metric_fn_mapping[Metric.ROUGE]
        rouge_result = rouge_fn(record)
        assert "rouge1" in rouge_result
        
        # Test factual consistency metric (will need mocked LLM responses)
        mock_responses = [
            Mock(content='{"questions": ["Q?"]}'),
            Mock(content='{"answers": ["Yes"]}'),
            Mock(content='{"answers": ["Yes"]}'),
            Mock(content='{"answers": ["No"]}')
        ]
        mock_llm.invoke.side_effect = mock_responses
        
        fc_fn = metric_fn_mapping[Metric.FACTUAL_CONSISTENCY]
        fc_result = fc_fn(record, llm=mock_llm, n=1)
        assert "human_factual_consistency" in fc_result
        assert "ai_factual_consistency" in fc_result