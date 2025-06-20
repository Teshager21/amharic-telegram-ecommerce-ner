"""
Main entrypoint for fine-tuning an Amharic NER model using Hugging Face Transformers.

Author: Teshager Admasu
Date: 2025-06-19

This script loads labeled CoNLL data, prepares the tokenizer and dataset,
fine-tunes the model with Trainer API, evaluates and saves the model.
"""

import logging
from pathlib import Path
import sys

from omegaconf import DictConfig
import hydra
from datasets import Dataset

from data.conll_loader import load_conll_data
from tokenizer_utils import prepare_tokenizer_and_align_labels
from trainer import NERTrainer
from transformers import AutoTokenizer, AutoModelForTokenClassification

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=cfg.logging.level,
        format="🎯 [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("🚀 Starting NER fine-tuning pipeline")

    # Load labeled CoNLL data
    data_path = Path(cfg.data.conll_file)
    logger.info(f"📥 Loading CoNLL data from: {data_path}")
    df = load_conll_data(data_path)

    # Load tokenizer
    logger.info(f"🔧 Loading tokenizer for model: {cfg.model.name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)

    # Build label2id mapping from unique labels in the dataset
    unique_labels = sorted(set(df["label"].unique()))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    logger.info(f"🔖 Label to ID mapping: {label2id}")

    # Group tokens and string labels by sentence
    texts = df.groupby("sentence_id")["token"].apply(list).tolist()
    labels_str = df.groupby("sentence_id")["label"].apply(list).tolist()

    # Convert string labels to integer IDs
    labels = [[label2id[label] for label in seq] for seq in labels_str]

    # Prepare tokenizer and align labels (integer IDs)
    logger.info("🔧 Preparing tokenizer and aligning labels")
    input_ids, aligned_labels = prepare_tokenizer_and_align_labels(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        label_all_tokens=cfg.training.label_all_tokens,
        max_length=cfg.model.max_length,
    )

    # Create Hugging Face Dataset object for trainer
    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "labels": aligned_labels,
        }
    )

    # Load model with proper label mapping
    logger.info(f"🔧 Loading model for token classification: {cfg.model.name_or_path}")
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model.name_or_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # Initialize trainer
    trainer = NERTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,  # Add eval dataset if available
        output_dir=Path(cfg.output_dir),
        training_args_dict={
            "num_train_epochs": cfg.training.epochs,
            "learning_rate": cfg.training.learning_rate,
            "per_device_train_batch_size": cfg.training.batch_size,
            "per_device_eval_batch_size": cfg.training.batch_size,
            "eval_strategy": cfg.training.eval_strategy,
            "seed": cfg.training.seed,
        },
    )

    # Train and evaluate
    trainer.train()
    trainer.evaluate()

    # Save model
    trainer.save_model()
    logger.info("✅ NER fine-tuning completed successfully.")


if __name__ == "__main__":
    main()
