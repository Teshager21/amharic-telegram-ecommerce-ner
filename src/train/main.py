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

    # Load train data
    train_data_path = Path(cfg.data.train_file)
    logger.info(f"📥 Loading training CoNLL data from: {train_data_path}")
    train_df = load_conll_data(train_data_path)

    # Load eval data if provided
    eval_df = None
    if "eval_file" in cfg.data and cfg.data.eval_file:
        eval_data_path = Path(cfg.data.eval_file)
        logger.info(f"📥 Loading evaluation CoNLL data from: {eval_data_path}")
        eval_df = load_conll_data(eval_data_path)
    else:
        logger.warning("⚠️ No evaluation dataset provided; evaluation will be skipped.")

    # Build label2id mapping
    unique_labels = sorted(set(train_df["label"].unique()))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    logger.info(f"🔖 Label to ID mapping: {label2id}")

    # Load tokenizer
    logger.info(f"🔧 Loading tokenizer for model: {cfg.model.name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)

    # Prepare HF dataset
    def prepare_dataset(df):
        texts = df.groupby("sentence_id")["token"].apply(list).tolist()
        labels_str = df.groupby("sentence_id")["label"].apply(list).tolist()
        labels = [[label2id[label] for label in seq] for seq in labels_str]
        input_ids, aligned_labels = prepare_tokenizer_and_align_labels(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            label_all_tokens=cfg.training.label_all_tokens,
            max_length=cfg.model.max_length,
        )
        return Dataset.from_dict(
            {
                "input_ids": input_ids,
                "labels": aligned_labels,
            }
        )

    train_dataset = prepare_dataset(train_df)
    eval_dataset = prepare_dataset(eval_df) if eval_df is not None else None

    # Load model
    logger.info(f"🔧 Loading model for token classification: {cfg.model.name_or_path}")
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model.name_or_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # Prepare training arguments dictionary
    training_args_dict = {
        "num_train_epochs": cfg.training.epochs,
        "learning_rate": cfg.training.learning_rate,
        "per_device_train_batch_size": cfg.training.batch_size,
        "per_device_eval_batch_size": cfg.training.batch_size,
        "seed": cfg.training.seed,
    }

    # ✅ Translate custom eval_strategy to HF's evaluation_strategy
    if eval_dataset is not None and "eval_strategy" in cfg.training:
        training_args_dict["evaluation_strategy"] = cfg.training.eval_strategy
        training_args_dict["load_best_model_at_end"] = True
        training_args_dict["metric_for_best_model"] = "f1"
        training_args_dict["greater_is_better"] = True

    # Initialize and run trainer
    trainer = NERTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=Path(cfg.output_dir),
        training_args_dict=training_args_dict,
    )

    trainer.train()
    if eval_dataset is not None:
        trainer.evaluate()

    trainer.save_model()
    logger.info("✅ NER fine-tuning completed successfully.")


if __name__ == "__main__":
    main()
