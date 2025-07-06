"""Tests for prompt templates."""

import pytest

from experiments.prompt_templates import (
    PromptTemplates,
    create_fine_tuning_prompt,
    create_basic_summary_prompt,
    create_word_count_prompt,
    create_subjective_length_prompt,
    create_sentence_based_prompt,
    create_alternative_word_count_prompt,
    format_training_example,
    get_prompt_for_config,
    PROMPT_CONFIGS
)


class TestPromptTemplates:
    """Test the PromptTemplates class."""
    
    def test_fine_tuning_template_exists(self):
        """Test that fine-tuning template is defined."""
        assert hasattr(PromptTemplates, 'FINE_TUNING_TEMPLATE')
        assert isinstance(PromptTemplates.FINE_TUNING_TEMPLATE, str)
        assert "{text}" in PromptTemplates.FINE_TUNING_TEMPLATE
        assert "{target_word_count}" in PromptTemplates.FINE_TUNING_TEMPLATE
    
    def test_response_template_exists(self):
        """Test that response template is defined."""
        assert hasattr(PromptTemplates, 'FINE_TUNING_RESPONSE_TEMPLATE')
        assert isinstance(PromptTemplates.FINE_TUNING_RESPONSE_TEMPLATE, str)
        assert "{summary}" in PromptTemplates.FINE_TUNING_RESPONSE_TEMPLATE
    
    def test_basic_summary_template_exists(self):
        """Test that basic summary template is defined."""
        assert hasattr(PromptTemplates, 'BASIC_SUMMARY_TEMPLATE')
        assert isinstance(PromptTemplates.BASIC_SUMMARY_TEMPLATE, str)
        assert "{instructions}" in PromptTemplates.BASIC_SUMMARY_TEMPLATE
    
    def test_word_count_template_exists(self):
        """Test that word count template is defined."""
        assert hasattr(PromptTemplates, 'WORD_COUNT_TEMPLATE')
        assert isinstance(PromptTemplates.WORD_COUNT_TEMPLATE, str)
        assert "{target_word_count}" in PromptTemplates.WORD_COUNT_TEMPLATE


class TestCreateFinetuningPrompt:
    """Test the create_fine_tuning_prompt function."""
    
    def test_create_fine_tuning_prompt_without_response(self):
        """Test creating fine-tuning prompt without response."""
        text = "This is a test article for summarization."
        target_word_count = 25
        
        prompt = create_fine_tuning_prompt(
            text=text,
            target_word_count=target_word_count,
            include_response=False
        )
        
        assert text in prompt
        assert str(target_word_count) in prompt
        assert "[WORD_COUNT]" in prompt
        assert "[END_WORD_COUNT]" in prompt
        assert "<|start_header_id|>system<|end_header_id|>" in prompt
        assert "<|start_header_id|>assistant<|end_header_id|>" in prompt
    
    def test_create_fine_tuning_prompt_with_response(self):
        """Test creating fine-tuning prompt with response."""
        text = "This is a test article for summarization."
        summary = "Test article summary."
        target_word_count = 25
        
        prompt = create_fine_tuning_prompt(
            text=text,
            target_word_count=target_word_count,
            summary=summary,
            include_response=True
        )
        
        assert text in prompt
        assert summary in prompt
        assert str(target_word_count) in prompt
        assert "<|eot_id|><|end_of_text|>" in prompt
    
    def test_create_fine_tuning_prompt_no_summary_with_response_flag(self):
        """Test that no response is added when summary is None even with include_response=True."""
        text = "Test text"
        target_word_count = 20
        
        prompt = create_fine_tuning_prompt(
            text=text,
            target_word_count=target_word_count,
            summary=None,
            include_response=True
        )
        
        assert text in prompt
        assert "<|eot_id|><|end_of_text|>" not in prompt


class TestCreateBasicSummaryPrompt:
    """Test the create_basic_summary_prompt function."""
    
    def test_create_basic_summary_prompt_no_instructions(self):
        """Test creating basic summary prompt without instructions."""
        text = "Test article content."
        
        messages = create_basic_summary_prompt(text)
        
        assert len(messages) == 2
        assert messages[0][0] == "system"
        assert messages[1][0] == "human"
        assert messages[1][1] == text
        assert "summary" in messages[0][1].lower()
    
    def test_create_basic_summary_prompt_with_instructions(self):
        """Test creating basic summary prompt with instructions."""
        text = "Test article content."
        instructions = "Make it concise and clear."
        
        messages = create_basic_summary_prompt(text, instructions)
        
        assert len(messages) == 2
        assert instructions in messages[0][1]
        assert text == messages[1][1]


class TestCreateWordCountPrompt:
    """Test the create_word_count_prompt function."""
    
    def test_create_word_count_prompt(self):
        """Test creating word count prompt."""
        text = "Sample text for summarization."
        target_word_count = 30
        
        messages = create_word_count_prompt(text, target_word_count)
        
        assert len(messages) == 2
        assert messages[0][0] == "system"
        assert messages[1][0] == "human"
        assert str(target_word_count) in messages[0][1]
        assert "exactly" in messages[0][1].lower()
        assert messages[1][1] == text


class TestCreateSubjectiveLengthPrompt:
    """Test the create_subjective_length_prompt function."""
    
    def test_create_subjective_length_prompt(self):
        """Test creating subjective length prompt."""
        text = "Article to summarize."
        length_instruction = "concise"
        
        messages = create_subjective_length_prompt(text, length_instruction)
        
        assert len(messages) == 2
        assert length_instruction in messages[0][1]
        assert messages[1][1] == text
    
    def test_create_subjective_length_prompt_brief(self):
        """Test creating subjective length prompt with 'brief'."""
        text = "Another article."
        length_instruction = "brief"
        
        messages = create_subjective_length_prompt(text, length_instruction)
        
        assert length_instruction in messages[0][1]


class TestCreateSentenceBasedPrompt:
    """Test the create_sentence_based_prompt function."""
    
    def test_create_sentence_based_prompt_one_sentence(self):
        """Test creating sentence-based prompt for one sentence."""
        text = "Text to summarize in one sentence."
        sentence_count = "one sentence long"
        
        messages = create_sentence_based_prompt(text, sentence_count)
        
        assert len(messages) == 2
        assert sentence_count in messages[0][1]
        assert messages[1][1] == text
    
    def test_create_sentence_based_prompt_two_sentences(self):
        """Test creating sentence-based prompt for two sentences."""
        text = "Text to summarize in two sentences."
        sentence_count = "two sentences or less"
        
        messages = create_sentence_based_prompt(text, sentence_count)
        
        assert sentence_count in messages[0][1]


class TestCreateAlternativeWordCountPrompt:
    """Test the create_alternative_word_count_prompt function."""
    
    def test_create_alternative_word_count_prompt_words(self):
        """Test creating alternative word count prompt for words."""
        text = "Text for word count control."
        target_count = 40
        
        messages = create_alternative_word_count_prompt(text, target_count, "words")
        
        assert len(messages) == 2
        assert f"exact {target_count} words" in messages[0][1]
        assert messages[1][1] == text
    
    def test_create_alternative_word_count_prompt_tokens(self):
        """Test creating alternative word count prompt for tokens."""
        text = "Text for token count control."
        target_count = 70
        
        messages = create_alternative_word_count_prompt(text, target_count, "tokens")
        
        assert f"exact {target_count} tokens" in messages[0][1]
    
    def test_create_alternative_word_count_prompt_invalid_type(self):
        """Test that invalid count_type raises ValueError."""
        text = "Test text."
        
        with pytest.raises(ValueError):
            create_alternative_word_count_prompt(text, 25, "invalid_type")


class TestFormatTrainingExample:
    """Test the format_training_example function."""
    
    def test_format_training_example(self):
        """Test formatting a training example."""
        sample = {
            "text": "Sample article text for training.",
            "summary": "Sample summary."
        }
        target_word_count = 15
        
        formatted = format_training_example(sample, target_word_count)
        
        assert sample["text"] in formatted
        assert sample["summary"] in formatted
        assert str(target_word_count) in formatted
        assert "[WORD_COUNT]" in formatted
        assert "[END_WORD_COUNT]" in formatted
        assert "<|eot_id|><|end_of_text|>" in formatted


class TestPromptConfigs:
    """Test predefined prompt configurations."""
    
    def test_prompt_configs_exist(self):
        """Test that PROMPT_CONFIGS is defined and contains expected configs."""
        assert isinstance(PROMPT_CONFIGS, dict)
        
        expected_configs = [
            "baseline", "word_count_25", "word_count_50", "word_count_75",
            "word_count_100", "word_count_150", "subjective_concise",
            "subjective_brief", "one_sentence", "two_sentences",
            "alt_word_count_30", "alt_token_count_70"
        ]
        
        for config in expected_configs:
            assert config in PROMPT_CONFIGS
    
    def test_prompt_config_structure(self):
        """Test that prompt configs have correct structure."""
        for config_name, config in PROMPT_CONFIGS.items():
            assert "template" in config
            assert "params" in config
            assert isinstance(config["template"], str)
            assert isinstance(config["params"], dict)
    
    def test_get_prompt_for_config_baseline(self):
        """Test getting prompt for baseline config."""
        text = "Test text for baseline."
        
        messages = get_prompt_for_config("baseline", text)
        
        assert len(messages) == 2
        assert messages[1][1] == text
    
    def test_get_prompt_for_config_word_count(self):
        """Test getting prompt for word count config."""
        text = "Test text for word count."
        
        messages = get_prompt_for_config("word_count_25", text)
        
        assert len(messages) == 2
        assert "25" in messages[0][1]
        assert "exactly" in messages[0][1].lower()
    
    def test_get_prompt_for_config_subjective(self):
        """Test getting prompt for subjective config."""
        text = "Test text for subjective length."
        
        messages = get_prompt_for_config("subjective_concise", text)
        
        assert len(messages) == 2
        assert "concise" in messages[0][1]
    
    def test_get_prompt_for_config_sentence_based(self):
        """Test getting prompt for sentence-based config."""
        text = "Test text for sentences."
        
        messages = get_prompt_for_config("one_sentence", text)
        
        assert len(messages) == 2
        assert "one sentence long" in messages[0][1]
    
    def test_get_prompt_for_config_alternative(self):
        """Test getting prompt for alternative config."""
        text = "Test text for alternative."
        
        messages = get_prompt_for_config("alt_word_count_30", text)
        
        assert len(messages) == 2
        assert "exact 30 words" in messages[0][1]
        
        messages = get_prompt_for_config("alt_token_count_70", text)
        
        assert "exact 70 tokens" in messages[0][1]
    
    def test_get_prompt_for_config_invalid(self):
        """Test that invalid config name raises ValueError."""
        with pytest.raises(ValueError):
            get_prompt_for_config("invalid_config", "test text")
    
    def test_get_prompt_for_config_unknown_template(self):
        """Test handling of unknown template type."""
        # This would require modifying PROMPT_CONFIGS, which we won't do
        # Instead, we test that all current configs work
        text = "Test text"
        
        for config_name in PROMPT_CONFIGS:
            try:
                messages = get_prompt_for_config(config_name, text)
                assert len(messages) == 2
                assert messages[1][1] == text
            except Exception as e:
                pytest.fail(f"Config {config_name} failed: {e}")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_text(self):
        """Test prompts with empty text."""
        text = ""
        
        # Should handle empty text gracefully
        messages = create_basic_summary_prompt(text)
        assert messages[1][1] == ""
        
        messages = create_word_count_prompt(text, 25)
        assert messages[1][1] == ""
    
    def test_long_text(self):
        """Test prompts with very long text."""
        text = "Long text. " * 1000  # Very long text
        
        messages = create_basic_summary_prompt(text)
        assert messages[1][1] == text
    
    def test_special_characters_in_text(self):
        """Test prompts with special characters."""
        text = "Text with special chars: <>&\"'\n\t"
        
        messages = create_basic_summary_prompt(text)
        assert messages[1][1] == text
    
    def test_zero_word_count_target(self):
        """Test with zero word count target."""
        text = "Test text"
        
        messages = create_word_count_prompt(text, 0)
        assert "0" in messages[0][1]
    
    def test_very_large_word_count_target(self):
        """Test with very large word count target."""
        text = "Test text"
        target = 10000
        
        messages = create_word_count_prompt(text, target)
        assert str(target) in messages[0][1]