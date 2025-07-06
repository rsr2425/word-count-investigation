"""Setup script for word-count-investigation experiments package."""

from setuptools import setup, find_packages

setup(
    name="word-count-investigation",
    version="0.1.0",
    description="Experiments for controlling word count in LLM-generated content",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "datasets>=2.14.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "langchain-openai>=0.1.0",
        "langchain-core>=0.1.0",
        "peft>=0.4.0",
        "trl>=0.7.0",
        "bitsandbytes>=0.41.0",
        "evaluate>=0.4.0",
        "rouge-score>=0.1.2",
        "wandb>=0.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pylint>=2.17.0",
            "isort>=5.12.0",
        ]
    },
    author="Ryan Rodriguez",
    author_email="rsr2425@example.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)