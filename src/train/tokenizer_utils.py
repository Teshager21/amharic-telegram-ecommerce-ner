# src/train/tokenizer_utils.py
"""
Tokenization and Label Alignment Utilities for NER Fine-Tuning.

Author: Teshager Admasu
Date: 2025-06-19
"""

from typing import List, Tuple
from transformers import PreTrainedTokenizerBase


def tokenize_and_align_labels(
    texts: List[List[str]],
    labels: List[List[str]],
    tokenizer: PreTrainedTokenizerBase,
    label_all_tokens: bool = True,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Tokenize input texts and align labels with the tokenized output for NER tasks.

    Args:
        texts (List[List[str]]): List of tokenized sentences (list of tokens).
        labels (List[List[str]]): Corresponding list of BIO labels for each token.
        tokenizer (PreTrainedTokenizerBase): Hugging Face tokenizer instance.
        label_all_tokens (bool): If True, label all subtokens with the original
        label; else, label only first subtoken.

    Returns:
        Tuple[List[List[int]], List[List[int]]]:
        Tokenized input IDs and aligned label IDs.
    """
    tokenized_inputs = tokenizer(
        texts,
        is_split_into_words=True,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens get ignored in loss
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)

    return tokenized_inputs["input_ids"], aligned_labels


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="💡 [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Tokenizer utils module ready to use.")
