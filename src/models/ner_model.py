# src/models/ner_model.py
"""
NER Model and Tokenizer Loader
Author: Teshager Admasu
Date: 2025-06-19
"""

import logging
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name_or_path: str, num_labels: int):
    """
    Load a pre-trained model and tokenizer for token classification (NER).

    Args:
        model_name_or_path (str): Hugging Face model ID or local path.
        num_labels (int): Number of unique entity labels.

    Returns:
        model, tokenizer
    """
    logger.info(f"📦 Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path, num_labels=num_labels
    )
    logger.info("✅ Model and tokenizer loaded successfully.")
    return model, tokenizer


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="🤖 [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="🔍 Load NER model and tokenizer")
    parser.add_argument(
        "--model", type=str, default="xlm-roberta-base", help="Hugging Face model ID"
    )
    parser.add_argument(
        "--num_labels", type=int, required=True, help="Number of entity labels"
    )

    args = parser.parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model, args.num_labels)
    print(model.config)
