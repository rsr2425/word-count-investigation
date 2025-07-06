"""Pytest configuration and shared fixtures for testing the experiments library."""

import pytest
import torch
from unittest.mock import Mock
from datasets import Dataset


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.eos_token_id = 128001
    tokenizer.eos_token = "<|end_of_text|>"
    tokenizer.pad_token = "<|end_of_text|>"
    tokenizer.padding_side = "right"
    tokenizer.vocab = {".": 13, "!": 0, "?": 30}
    
    def mock_convert_tokens_to_ids(token):
        return tokenizer.vocab.get(token, -1)
    
    def mock_decode(input_ids, skip_special_tokens=False):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        # Simple mock: return a string with length proportional to input
        return " ".join(["word"] * (len(input_ids) // 10 + 1))
    
    def mock_batch_decode(input_ids, skip_special_tokens=False):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        results = []
        for seq in input_ids:
            results.append(mock_decode([seq], skip_special_tokens))
        return results
    
    def mock_tokenize(text, return_tensors=None):
        # Simple mock: each word becomes ~3 tokens
        words = text.split()
        token_count = len(words) * 3
        input_ids = list(range(token_count))
        
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([input_ids])}
        return {"input_ids": input_ids}
    
    tokenizer.convert_tokens_to_ids = mock_convert_tokens_to_ids
    tokenizer.decode = mock_decode
    tokenizer.batch_decode = mock_batch_decode
    tokenizer.__call__ = mock_tokenize
    
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.device = torch.device("cpu")
    
    def mock_generate(input_ids, **kwargs):
        # Simple mock: return input_ids with some additional tokens
        batch_size = input_ids.shape[0]
        additional_tokens = torch.randint(0, 1000, (batch_size, 20))
        return torch.cat([input_ids, additional_tokens], dim=1)
    
    model.generate = mock_generate
    return model


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    
    def mock_invoke(messages):
        # Extract the human message and create a simple response
        human_msg = ""
        for role, content in messages:
            if role == "human":
                human_msg = content
                break
        
        # Simple mock: return a summary that's ~10% of input length
        words = human_msg.split()
        summary_length = max(5, len(words) // 10)
        summary = " ".join(["summary"] * summary_length) + "."
        
        response = Mock()
        response.content = summary
        return response
    
    llm.invoke = mock_invoke
    return llm


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = {
        "text": [
            "This is a sample article with multiple sentences. It contains information about various topics. The article is designed to test summarization functionality.",
            "Another sample article for testing purposes. This one also has multiple sentences and covers different topics.",
        ],
        "summary": [
            "Sample article about various topics.",
            "Another test article covering different topics.",
        ]
    }
    return Dataset.from_dict(data)


@pytest.fixture
def sample_record():
    """Create a sample record for testing."""
    return {
        "text": "This is a test article with multiple sentences for testing purposes. It should be summarized effectively.",
        "summary": "Test article for summarization."
    }


@pytest.fixture 
def mock_rouge_score():
    """Create a mock ROUGE score evaluator."""
    rouge = Mock()
    
    def mock_compute(predictions, references):
        # Mock computation - return fixed scores
        return {
            "rouge1": 0.5,
            "rouge2": 0.3,
            "rougeL": 0.4,
            "rougeLsum": 0.45
        }
    
    rouge.compute = mock_compute
    return rouge


@pytest.fixture(autouse=True)
def mock_evaluate_load(monkeypatch):
    """Mock the evaluate.load function to avoid downloading models during tests."""
    def mock_load(metric_name):
        if metric_name == "rouge":
            rouge = Mock()
            rouge.compute = Mock(return_value={
                "rouge1": 0.5,
                "rouge2": 0.3, 
                "rougeL": 0.4,
                "rougeLsum": 0.45
            })
            return rouge
        return Mock()
    
    import evaluate
    monkeypatch.setattr(evaluate, "load", mock_load)


@pytest.fixture(autouse=True) 
def disable_wandb(monkeypatch):
    """Disable wandb logging during tests."""
    wandb_mock = Mock()
    wandb_mock.init = Mock()
    wandb_mock.log = Mock()
    wandb_mock.finish = Mock()
    wandb_mock.Table = Mock()
    
    monkeypatch.setattr("wandb.init", wandb_mock.init)
    monkeypatch.setattr("wandb.log", wandb_mock.log)
    monkeypatch.setattr("wandb.finish", wandb_mock.finish)
    monkeypatch.setattr("wandb.Table", wandb_mock.Table)


# Mock transformers components to avoid loading large models
@pytest.fixture(autouse=True)
def mock_transformers(monkeypatch):
    """Mock transformers components to avoid loading models during tests."""
    
    def mock_from_pretrained(*_args, **_kwargs):
        return Mock()
    
    # Mock AutoModel and AutoTokenizer
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", mock_from_pretrained)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda x: Mock())
    
    # Mock BitsAndBytesConfig
    monkeypatch.setattr("transformers.BitsAndBytesConfig", Mock)