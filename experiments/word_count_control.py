from typing import Any, Dict, List, Tuple, Union, Optional
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


def count_words(text: str) -> int:
    """Count the number of words in a given text.
    
    Args:
        text: The input text to count words in
        
    Returns:
        The number of words in the text
    """
    return len(text.split())


class WordCountControlRunnable(Runnable):
    """A runnable that controls word count through iterative refinement.
    
    This class implements a runnable that can be used to generate text with a target
    word count by iteratively refining the output until it falls within the specified
    tolerance of the target word count.
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        word_count_target: int = 25,
        tolerance: int = 10,
        max_revision_attempts: int = 5,
    ):
        """Initialize the WordCountControlRunnable.
        
        Args:
            llm: The language model to use for generation
            word_count_target: The target number of words for the output
            tolerance: The acceptable deviation from the target word count
            max_revision_attempts: Maximum number of refinement attempts
        """
        self.llm = llm
        self.word_count_target = word_count_target
        self.tolerance = tolerance
        self.max_revision_attempts = max_revision_attempts
    
    def _count_words(self, text: str) -> int:
        """Utility method to count words in text."""
        return count_words(text)
    
    def _is_within_tolerance(self, text: str) -> bool:
        """Check if the word count of text is within tolerance of target."""
        word_count = self._count_words(text)
        return abs(word_count - self.word_count_target) <= self.tolerance
    
    def invoke(self, input: Union[str, Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """Generate text with controlled word count.
        
        Args:
            input: Either a string or dict containing the input text
            
        Returns:
            A dictionary containing:
                - final_summary: The generated text with controlled word count
                - attempts: Number of refinement attempts made
        """
        # Extract the input text
        if isinstance(input, str):
            sample_text = input
        else:
            sample_text = input.get("sample_text", input.get("text", ""))
        
        # Prepare initial messages
        messages = [
            ("system", "You are a helpful summary chatbot. Summarize the content provided by the user."),
            ("human", sample_text),
        ]
        
        # Initialize variables for refinement loop
        attempt = 0
        ai_summary = None
        
        # Iterative refinement loop
        while attempt < self.max_revision_attempts and (
            ai_summary is None or not self._is_within_tolerance(ai_summary)
        ):
            attempt += 1
            
            # Add word count instructions after first attempt
            if attempt > 1:
                current_count = self._count_words(ai_summary)
                direction = "shorter" if current_count > self.word_count_target else "longer"
                messages.append(("ai", ai_summary))
                messages.append((
                    "human",
                    f"Please make this summary {direction}. "
                    f"Aim for around {self.word_count_target} words. "
                    f"Current length: {current_count} words."
                ))
            
            # Generate response
            ai_msg = self.llm.invoke(messages)
            ai_summary = ai_msg.content
        
        return {
            "final_summary": ai_summary,
            "attempts": attempt
        }


def create_word_count_controlled_chain(
    llm: ChatOpenAI,
    word_count_target: int = 25,
    tolerance: int = 10,
    max_revision_attempts: int = 5,
) -> Runnable:
    """Create a chain that controls word count through iterative refinement.
    
    Args:
        llm: The language model to use for generation
        word_count_target: The target number of words for the output
        tolerance: The acceptable deviation from the target word count
        max_revision_attempts: Maximum number of refinement attempts
        
    Returns:
        A runnable chain that can be used with LangChain
    """
    word_count_controller = WordCountControlRunnable(
        llm=llm,
        word_count_target=word_count_target,
        tolerance=tolerance,
        max_revision_attempts=max_revision_attempts,
    )
    return llm | StrOutputParser() | word_count_controller
