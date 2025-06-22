"""
NER Model Training Module with Hugging Face Trainer and MLflow integration.

Author: Teshager Admasu
Date: 2025-06-20
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import mlflow
import datasets
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

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

        # Base/default training arguments
        default_args = {
            "output_dir": str(self.output_dir),
            "eval_strategy": "epoch",  # <-- Use eval_strategy here
            "save_strategy": "epoch",
            "logging_strategy": "steps",
            "logging_steps": 100,
            "save_total_limit": 2,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "seed": 42,
        }

        # Include evaluation-related settings only if eval set is provided
        if eval_dataset is not None:
            default_args.update(
                {
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "f1",
                    "greater_is_better": True,
                }
            )

        # Update default_args with any user-provided args
        if training_args_dict:
            # Directly update without converting keys
            default_args.update(training_args_dict)

        self.training_args = TrainingArguments(**default_args)
        self.data_collator = DataCollatorForTokenClassification(tokenizer)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics if eval_dataset else None,
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
        id2label = self.model.config.id2label

        # Remove ignored index (-100) and convert to string labels
        true_labels = [
            [id2label[int(label)] for label in label_seq if label != -100]
            for label_seq in labels
        ]
        true_predictions = [
            [
                id2label[int(pred)]
                for pred, label in zip(pred_seq, label_seq)
                if label != -100
            ]
            for pred_seq, label_seq in zip(predictions, labels)
        ]

        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)
        accuracy = accuracy_score(true_labels, true_predictions)

        logger.info(
            f"📊 Eval — Precision: {precision:.4f}, Recall: "
            f"{recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}"
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    def train(self):
        """
        Train the model and log with MLflow.
        """
        mlflow.set_experiment(
            "NER_Training"
        )  # 🔧 Use a valid experiment name, not "mlruns"
        with mlflow.start_run():
            mlflow.log_param("epochs", self.training_args.num_train_epochs)
            mlflow.log_param("learning_rate", self.training_args.learning_rate)
            mlflow.log_param(
                "train_batch_size", self.training_args.per_device_train_batch_size
            )

            logger.info("🚀 Starting training...")
            train_result = self.trainer.train()

            # Save model and tokenizer explicitly to output_dir
            self.trainer.save_model(str(self.output_dir))
            self.tokenizer.save_pretrained(str(self.output_dir))

            # Log model directory as MLflow artifact
            mlflow.log_artifacts(str(self.output_dir), artifact_path="model")

            metrics = train_result.metrics
            mlflow.log_metrics(metrics)
            logger.info(f"✅ Training completed. Metrics: {metrics}")
            return metrics

    def evaluate(self):
        """
        Evaluate the model on the evaluation dataset.
        """
        if self.eval_dataset is None:
            logger.warning("⚠️ No evaluation dataset provided, skipping evaluation.")
            return None

        logger.info("🧪 Running evaluation...")
        metrics = self.trainer.evaluate()
        logger.info(f"📊 Evaluation metrics: {metrics}")
        return metrics

    def save_model(self):
        """
        Save model to output directory.
        """
        logger.info(f"💾 Saving model to {self.output_dir}")
        self.trainer.save_model(self.output_dir)
        # Log model directory as artifact
        mlflow.log_artifacts(self.output_dir, artifact_path="ner_model")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="🔥 [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Train NER model with Hugging Face Trainer + MLflow"
    )
    parser.add_argument(
        "--train_file", type=str, required=True, help="Path to training dataset"
    )
    parser.add_argument("--eval_file", type=str, help="Path to evaluation dataset")
    parser.add_argument(
        "--model_name", type=str, default="xlm-roberta-base", help="Model name"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/ner", help="Output directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    args = parser.parse_args()
    logger.warning(
        "⚠️ This CLI mode is a template and not integrated with Hydra config."
    )
