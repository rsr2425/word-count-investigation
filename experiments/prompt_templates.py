"""Prompt templates for different word count control approaches."""

from typing import Dict, Any, Optional


class PromptTemplates:
    """Collection of prompt templates for word count experiments."""
    
    # Template for fine-tuning with word count targets
    FINE_TUNING_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful summary chatbot. Summarize the text provided by the user.
The word count of your response should be less than or equal to the value specified in the WORD_COUNT/END_WORD_COUNT block.<|eot_id|><|start_header_id|>user<|end_header_id|>
{text}

[WORD_COUNT]
{target_word_count}
[END_WORD_COUNT]<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    # Response template for fine-tuning
    FINE_TUNING_RESPONSE_TEMPLATE = """\
{summary}<|eot_id|><|end_of_text|>"""
    
    # Template for basic summarization without word count
    BASIC_SUMMARY_TEMPLATE = """\
You are a helpful summary chatbot. Summarize the content provided by the user. {instructions}"""
    
    # Template for explicit word count instructions
    WORD_COUNT_TEMPLATE = """\
You are a helpful summary chatbot. Summarize the content provided by the user in exactly {target_word_count} words."""
    
    # Template for subjective length instructions
    SUBJECTIVE_LENGTH_TEMPLATE = """\
You are a helpful summary chatbot. Summarize the content provided by the user. Make the summary {length_instruction}."""
    
    # Template for sentence-based length control
    SENTENCE_BASED_TEMPLATE = """\
You are a helpful summary chatbot. Summarize the content provided by the user. Make the summary {sentence_count}."""


def create_fine_tuning_prompt(
    text: str, 
    target_word_count: int, 
    summary: Optional[str] = None,
    include_response: bool = True
) -> str:
    """Create a prompt for fine-tuning with word count control.
    
    Args:
        text: The input text to summarize
        target_word_count: The target word count for the summary
        summary: The expected summary (for training data)
        include_response: Whether to include the response template
        
    Returns:
        Formatted prompt string
    """
    prompt = PromptTemplates.FINE_TUNING_TEMPLATE.format(
        text=text,
        target_word_count=target_word_count
    )
    
    if include_response and summary:
        prompt += PromptTemplates.FINE_TUNING_RESPONSE_TEMPLATE.format(summary=summary)
    
    return prompt


def create_basic_summary_prompt(
    text: str,
    instructions: str = ""
) -> list:
    """Create a basic summarization prompt.
    
    Args:
        text: The input text to summarize
        instructions: Additional instructions for summarization
        
    Returns:
        List of message tuples for LangChain
    """
    system_message = PromptTemplates.BASIC_SUMMARY_TEMPLATE.format(
        instructions=instructions
    ).strip()
    
    return [
        ("system", system_message),
        ("human", text),
    ]


def create_word_count_prompt(
    text: str,
    target_word_count: int
) -> list:
    """Create a prompt with explicit word count instructions.
    
    Args:
        text: The input text to summarize
        target_word_count: The target number of words
        
    Returns:
        List of message tuples for LangChain
    """
    system_message = PromptTemplates.WORD_COUNT_TEMPLATE.format(
        target_word_count=target_word_count
    )
    
    return [
        ("system", system_message),
        ("human", text),
    ]


def create_subjective_length_prompt(
    text: str,
    length_instruction: str
) -> list:
    """Create a prompt with subjective length instructions.
    
    Args:
        text: The input text to summarize
        length_instruction: Subjective length instruction (e.g., "concise", "brief")
        
    Returns:
        List of message tuples for LangChain
    """
    system_message = PromptTemplates.SUBJECTIVE_LENGTH_TEMPLATE.format(
        length_instruction=length_instruction
    )
    
    return [
        ("system", system_message),
        ("human", text),
    ]


def create_sentence_based_prompt(
    text: str,
    sentence_count: str
) -> list:
    """Create a prompt with sentence-based length control.
    
    Args:
        text: The input text to summarize
        sentence_count: Sentence count instruction (e.g., "one sentence long", "two sentences or less")
        
    Returns:
        List of message tuples for LangChain
    """
    system_message = PromptTemplates.SENTENCE_BASED_TEMPLATE.format(
        sentence_count=sentence_count
    )
    
    return [
        ("system", system_message),
        ("human", text),
    ]


def create_alternative_word_count_prompt(
    text: str,
    target_count: int,
    count_type: str = "words"
) -> list:
    """Create a prompt with alternative word/token count phrasing.
    
    Args:
        text: The input text to summarize
        target_count: The target count
        count_type: Type of count ("words" or "tokens")
        
    Returns:
        List of message tuples for LangChain
    """
    if count_type == "words":
        instruction = f"Use exact {target_count} words in your summary."
    elif count_type == "tokens":
        instruction = f"Use exact {target_count} tokens in your summary."
    else:
        raise ValueError("count_type must be 'words' or 'tokens'")
    
    system_message = f"You are a helpful summary chatbot. Summarize the content provided by the user. {instruction}"
    
    return [
        ("system", system_message),
        ("human", text),
    ]


def format_training_example(
    sample: Dict[str, Any],
    target_word_count: int
) -> str:
    """Format a training example for fine-tuning.
    
    Args:
        sample: Dictionary containing 'text' and 'summary' keys
        target_word_count: Target word count for the summary
        
    Returns:
        Formatted training example
    """
    return create_fine_tuning_prompt(
        text=sample["text"],
        target_word_count=target_word_count,
        summary=sample["summary"],
        include_response=True
    )


# Predefined prompt configurations for common experiments
PROMPT_CONFIGS = {
    "baseline": {
        "template": "basic_summary",
        "params": {"instructions": ""}
    },
    "word_count_25": {
        "template": "word_count",
        "params": {"target_word_count": 25}
    },
    "word_count_50": {
        "template": "word_count", 
        "params": {"target_word_count": 50}
    },
    "word_count_75": {
        "template": "word_count",
        "params": {"target_word_count": 75}
    },
    "word_count_100": {
        "template": "word_count",
        "params": {"target_word_count": 100}
    },
    "word_count_150": {
        "template": "word_count",
        "params": {"target_word_count": 150}
    },
    "subjective_concise": {
        "template": "subjective_length",
        "params": {"length_instruction": "concise"}
    },
    "subjective_brief": {
        "template": "subjective_length",
        "params": {"length_instruction": "brief"}
    },
    "one_sentence": {
        "template": "sentence_based",
        "params": {"sentence_count": "one sentence long"}
    },
    "two_sentences": {
        "template": "sentence_based",
        "params": {"sentence_count": "two sentences or less"}
    },
    "alt_word_count_30": {
        "template": "alternative",
        "params": {"target_count": 30, "count_type": "words"}
    },
    "alt_token_count_70": {
        "template": "alternative",
        "params": {"target_count": 70, "count_type": "tokens"}
    }
}


def get_prompt_for_config(config_name: str, text: str) -> list:
    """Get a prompt based on a predefined configuration.
    
    Args:
        config_name: Name of the configuration to use
        text: The input text to summarize
        
    Returns:
        List of message tuples for LangChain
        
    Raises:
        ValueError: If config_name is not recognized
    """
    if config_name not in PROMPT_CONFIGS:
        raise ValueError(f"Unknown prompt config: {config_name}. Available: {list(PROMPT_CONFIGS.keys())}")
    
    config = PROMPT_CONFIGS[config_name]
    template = config["template"]
    params = config["params"]
    
    if template == "basic_summary":
        return create_basic_summary_prompt(text, **params)
    elif template == "word_count":
        return create_word_count_prompt(text, **params)
    elif template == "subjective_length":
        return create_subjective_length_prompt(text, **params)
    elif template == "sentence_based":
        return create_sentence_based_prompt(text, **params)
    elif template == "alternative":
        return create_alternative_word_count_prompt(text, **params)
    else:
        raise ValueError(f"Unknown template: {template}")