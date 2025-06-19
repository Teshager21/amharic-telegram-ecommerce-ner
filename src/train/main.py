# src/train/main.py
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

from conll_loader import load_conll_data
from tokenizer_utils import prepare_tokenizer_and_align_labels
from trainer import NERTrainer

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

    # Prepare tokenizer and align labels
    logger.info(f"🔧 Preparing tokenizer for model: {cfg.model.name_or_path}")
    tokenizer, label_all_tokens, encoded_dataset = prepare_tokenizer_and_align_labels(
        df,
        model_name=cfg.model.name_or_path,
        max_length=cfg.model.max_length,
        label_all_tokens=cfg.training.label_all_tokens,
    )

    # Initialize trainer
    trainer = NERTrainer(
        model_name=cfg.model.name_or_path,
        tokenizer=tokenizer,
        dataset=encoded_dataset,
        output_dir=Path(cfg.output_dir),
        num_train_epochs=cfg.training.epochs,
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        evaluation_strategy=cfg.training.evaluation_strategy,
        seed=cfg.training.seed,
    )

    # Train and evaluate
    trainer.train()
    trainer.evaluate()

    # Save model
    trainer.save_model()
    logger.info("✅ NER fine-tuning completed successfully.")


if __name__ == "__main__":
    main()
