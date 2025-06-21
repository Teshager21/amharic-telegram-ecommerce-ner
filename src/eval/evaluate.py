# src/eval/evaluate.py
"""
Evaluate a fine-tuned NER model on a labeled dataset with MLflow integration.
Author: Teshager Admasu
Date: 2025-06-19
"""

import logging
from pathlib import Path
from typing import List

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

logging.basicConfig(level=logging.INFO, format="📊 [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def tokenize_and_align_labels(df: pd.DataFrame, tokenizer, label2id: dict):
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
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]])
            previous_word_idx = word_idx
        encoded_labels.append(label_ids)

    encodings["labels"] = encoded_labels
    return Dataset.from_dict(encodings)


def evaluate(model_path: str, data_path: str, label_list: List[str]):
    from src.data.conll_loader import load_conll_data

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    logger.info("📥 Loading dataset and model...")
    df = load_conll_data(data_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    dataset = tokenize_and_align_labels(df, tokenizer, label2id)

    all_preds = []
    all_labels = []

    logger.info("🔍 Running evaluation...")
    for batch in torch.utils.data.DataLoader(dataset, batch_size=16):
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        for pred, label in zip(predictions, batch["labels"]):
            true_labels = []
            pred_labels = []
            for p, l in zip(pred, label):
                if l != -100:
                    true_labels.append(id2label[l.item()])
                    pred_labels.append(id2label[p.item()])
            all_preds.append(pred_labels)
            all_labels.append(true_labels)

    logger.info("✅ Evaluation complete. Classification report:")
    report = classification_report(all_labels, all_preds)
    print(report)

    # MLflow logging
    with mlflow.start_run(run_name="NER_Evaluation"):
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("data_path", data_path)

        mlflow.log_metric("precision", precision_score(all_labels, all_preds))
        mlflow.log_metric("recall", recall_score(all_labels, all_preds))
        mlflow.log_metric("f1", f1_score(all_labels, all_preds))
        mlflow.log_metric("accuracy", accuracy_score(all_labels, all_preds))

        output_path = Path("eval_report.txt")
        output_path.write_text(report)
        mlflow.log_metric("f1", f1_score(...))
        mlflow.log_artifact(str(output_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="📊 Evaluate NER model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to CoNLL test data"
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
        model_path=args.model_path, data_path=args.data_path, label_list=LABEL_LIST
    )
