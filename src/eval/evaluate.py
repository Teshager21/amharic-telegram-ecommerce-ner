"""
Evaluate a fine-tuned NER model on a labeled dataset with optional MLflow integration.

Author: Teshager Admasu
Date: 2025-06-19
"""

import sys
import pathlib
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import mlflow
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from src.data.conll_loader import load_conll_data

# Add project root folder to sys.path to allow imports to work
project_root = pathlib.Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="📊 [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def tokenize_and_align_labels(df: pd.DataFrame, tokenizer, label2id: dict) -> Dataset:
    grouped = df.groupby("sentence_id").agg(list)
    texts = grouped["token"].tolist()
    labels = grouped["label"].tolist()

    encodings = tokenizer(
        texts,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
    )

    encoded_labels = []
    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[label[word_idx]])
        encoded_labels.append(label_ids)

    encodings["labels"] = encoded_labels
    return Dataset.from_dict(encodings)


def evaluate(
    model_path: str,
    data_path: str,
    label_list: Optional[List[str]] = None,
    batch_size: int = 16,
    use_mlflow: bool = True,
    output_report_path: Optional[str] = None,
):
    """
    Evaluate a fine-tuned NER model on labeled data.

    Args:
        model_path: Path to the fine-tuned model directory.
        data_path: Path to labeled test data in CoNLL format.
        label_list: Optional list of label strings in order.
        If None, inferred from model.
        batch_size: Batch size for evaluation dataloader.
        use_mlflow: Whether to log results to MLflow.
        Set False in environments without MLflow.
        output_report_path: Optional path to save the
        classification report as a text file.
    """
    from data.conll_loader import (
        load_conll_data,
    )  # local import for notebook compatibility

    logger.info("📥 Loading dataset and model...")
    df = load_conll_data(data_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    if label_list is None:
        label_list = [
            model.config.id2label[i] for i in range(len(model.config.id2label))
        ]
        logger.info(f"🔖 Using label list from model config: {label_list}")

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    logger.info(f"Model config id2label: {model.config.id2label}")
    logger.info(f"Using label_list: {label_list}")

    dataset = tokenize_and_align_labels(df, tokenizer, label2id)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    all_preds = []
    all_labels = []

    logger.info("🔍 Running evaluation...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    first_batch_printed = False

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        if not first_batch_printed:
            true_labels = batch["labels"][0]
            pred_labels = predictions[0]

            true_label_names = [
                id2label[label_id.item()]
                for label_id in true_labels
                if label_id.item() != -100
            ]
            pred_label_names = [
                id2label[pred_id.item()]
                for pred_id, label_id in zip(pred_labels, true_labels)
                if label_id.item() != -100
            ]

            print("===== DEBUG: First batch sample =====")
            print("True labels:     ", true_label_names)
            print("Predicted labels:", pred_label_names)
            print("=====================================")
            first_batch_printed = True

        for pred_tensor, label_tensor in zip(predictions, batch["labels"]):
            pred_seq = []
            true_seq = []
            for pred_id, label_id in zip(pred_tensor, label_tensor):
                if label_id != -100:
                    true_seq.append(id2label[label_id.item()])
                    pred_seq.append(id2label[pred_id.item()])
            all_preds.append(pred_seq)
            all_labels.append(true_seq)

    report = classification_report(all_labels, all_preds)
    logger.info("✅ Evaluation complete. Classification report:\n%s", report)
    print(report)

    if output_report_path:
        output_path = Path(output_report_path)
        output_path.write_text(report)
        logger.info(f"📝 Report saved to {output_report_path}")

    if use_mlflow:
        try:
            mlflow.set_experiment("NER_Evaluation_Experiment")
            with mlflow.start_run(run_name="NER_Evaluation"):
                mlflow.log_param("model_path", model_path)
                mlflow.log_param("data_path", data_path)
                mlflow.log_metric("precision", precision_score(all_labels, all_preds))
                mlflow.log_metric("recall", recall_score(all_labels, all_preds))
                mlflow.log_metric("f1", f1_score(all_labels, all_preds))
                mlflow.log_metric("accuracy", accuracy_score(all_labels, all_preds))
                if output_report_path:
                    mlflow.log_artifact(str(output_report_path))
        except Exception as e:
            logger.warning(f"⚠️ MLflow logging failed: {e}")


def evaluate_and_return_preds(
    model_path: str,
    data_path: str,
    label_list: Optional[List[str]] = None,
    batch_size: int = 16,
    return_tokens: bool = False,
):
    """
    Evaluate a fine-tuned NER model and return predictions and labels for CSV export.

    Returns:
        List[Dict] with keys: 'tokens', 'true_labels', 'pred_labels'
    """

    logger.info("📥 Loading dataset and model for prediction export...")
    df = load_conll_data(data_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    if label_list is None:
        label_list = [
            model.config.id2label[i] for i in range(len(model.config.id2label))
        ]
        logger.info(f"🔖 Using label list from model config: {label_list}")

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    dataset = tokenize_and_align_labels(df, tokenizer, label2id)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    results = []
    sentence_tokens = df.groupby("sentence_id")["token"].apply(list).tolist()
    sent_idx = 0

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        for b in range(len(batch["input_ids"])):
            true_seq = []
            pred_seq = []
            # word_ids = batch["input_ids"][b]
            labels = batch["labels"][b]
            preds = predictions[b]

            for pred_id, label_id in zip(preds, labels):
                if label_id.item() != -100:
                    true_seq.append(id2label[label_id.item()])
                    pred_seq.append(id2label[pred_id.item()])

            results.append(
                {
                    "tokens": sentence_tokens[sent_idx],
                    "true_labels": true_seq,
                    "pred_labels": pred_seq,
                }
            )
            sent_idx += 1

    logger.info("📦 Prediction export complete.")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="📊 Evaluate NER model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to CoNLL test data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--no_mlflow",
        action="store_true",
        help="Disable MLflow logging (useful for notebook or non-MLflow envs)",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="eval_report.txt",
        help="Path to save evaluation report",
    )
    args = parser.parse_args()

    LABEL_LIST = [
        "O",
        "B-PRODUCT",
        "I-PRODUCT",
        "B-PRICE",
        "I-PRICE",
        "B-LOC",
        "I-LOC",
    ]

    evaluate(
        model_path=args.model_path,
        data_path=args.data_path,
        label_list=LABEL_LIST,
        batch_size=args.batch_size,
        use_mlflow=not args.no_mlflow,
        output_report_path=args.output_report,
    )
