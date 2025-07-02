from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from .word_count_control import count_words, create_word_count_controlled_chain


def summarize(
    record: Dict[str, Any],
    llm: ChatOpenAI,
    word_count_target: Optional[int] = None,
    otherinstructions: Optional[str] = None,
    tolerance: int = 10,
    max_revision_attempts: int = 5,
) -> Dict[str, Any]:
    """Generate a summary of the input text with optional word count control.
    
    Args:
        record: Dictionary containing 'text' and 'summary' keys
        llm: The language model to use for generation
        word_count_target: Target number of words for the summary (optional)
        otherinstructions: Additional instructions for the summarization
        tolerance: Allowed deviation from word_count_target (if specified)
        max_revision_attempts: Maximum refinement attempts for word count control
        
    Returns:
        A dictionary containing:
            - text_word_count: Word count of input text
            - summary_word_count: Word count of reference summary
            - ai_summary: Generated summary
            - ai_summary_word_count: Word count of generated summary
            - total_model_calls: Number of model calls made (only when word_count_target is used)
    """
    # Prepare system message
    system_message = "You are a helpful summary chatbot. "
    if word_count_target is not None:
        system_message += f"Summarize the content in about {word_count_target} words. "
    if otherinstructions:
        system_message += otherinstructions
    
    # Create the appropriate chain based on whether we need word count control
    if word_count_target is not None:
        # Use word count controlled chain
        chain = create_word_count_controlled_chain(
            llm=llm,
            word_count_target=word_count_target,
            tolerance=tolerance,
            max_revision_attempts=max_revision_attempts,
        )
        
        # Run the chain
        result = chain.invoke({"sample_text": record['text']})
        
        return {
            'text_word_count': count_words(record['text']),
            'summary_word_count': count_words(record['summary']),
            'ai_summary': result['final_summary'],
            'ai_summary_word_count': count_words(result['final_summary']),
            'total_model_calls': result['attempts']
        }
    else:
        # Simple summarization without word count control
        messages = [
            ("system", system_message.strip()),
            ("human", record['text']),
        ]
        
        ai_summary = llm.invoke(messages).content
        
        return {
            'text_word_count': count_words(record['text']),
            'summary_word_count': count_words(record['summary']),
            'ai_summary': ai_summary,
            'ai_summary_word_count': count_words(ai_summary),
            'total_model_calls': 1
        }
