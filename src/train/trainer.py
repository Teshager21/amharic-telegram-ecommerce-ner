# src/train/trainer.py
"""
NER Model Training Module with Hugging Face Trainer and MLflow integration.
Author: Teshager Admasu
Date: 2025-06-19
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import mlflow
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
import datasets

logger = logging.getLogger(__name__)


class NERTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: datasets.Dataset,
        eval_dataset: Optional[datasets.Dataset] = None,
        output_dir: Path = Path("models"),
        training_args_dict: Optional[Dict[str, Any]] = None,
        mlflow_experiment_name: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        default_args = {
            "output_dir": str(self.output_dir),
            "evaluation_strategy": "epoch",  # ✅ added explicitly
            "save_strategy": "epoch",  # ✅ added explicitly
            "logging_strategy": "steps",
            "logging_steps": 100,
            "save_total_limit": 2,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "seed": 42,
        }

        if training_args_dict:
            default_args.update(training_args_dict)

        # Optional debugging: warn on unsupported keys
        from inspect import signature

        unsupported = [
            k for k in default_args if k not in signature(TrainingArguments).parameters
        ]
        if unsupported:
            logger.warning(f"⚠️ Unsupported TrainingArguments keys: {unsupported}")

        self.training_args = TrainingArguments(**default_args)
        self.data_collator = DataCollatorForTokenClassification(tokenizer)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.mlflow_experiment_name = mlflow_experiment_name or "NER_Training"

    def compute_metrics(self, p):
        from seqeval.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        predictions, labels = p
        predictions = predictions.argmax(-1)

        true_labels = [
            [label for label in label_seq if label != -100] for label_seq in labels
        ]
        true_predictions = [
            [p for p, l in zip(pred_seq, label_seq) if l != -100]
            for pred_seq, label_seq in zip(predictions, labels)
        ]

        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)
        accuracy = accuracy_score(true_labels, true_predictions)

        logger.info(
            f"Evaluation — Precision: {precision:.4f}, Recall: {recall:.4f}, "
            f"F1: {f1:.4f}, Accuracy: {accuracy:.4f}"
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    def train(self):
        mlflow.set_experiment(self.mlflow_experiment_name)
        with mlflow.start_run():
            mlflow.log_params(vars(self.training_args))
            logger.info("🚀 Starting training...")

            train_result = self.trainer.train()
            self.trainer.save_model()
            mlflow.log_artifacts(str(self.output_dir), artifact_path="model")

            metrics = train_result.metrics
            mlflow.log_metrics(metrics)

            logger.info(f"✅ Training finished. Metrics: {metrics}")
            return metrics

    def evaluate(self):
        logger.info("🧪 Running evaluation...")
        metrics = self.trainer.evaluate()
        logger.info(f"📊 Evaluation metrics: {metrics}")
        return metrics

    def save_model(self):
        logger.info(f"💾 Saving model to {self.output_dir}")
        self.trainer.save_model(self.output_dir)
