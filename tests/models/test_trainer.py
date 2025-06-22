import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
from datasets import Dataset
from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizerBase

from train.trainer import NERTrainer  # Adjust if your module path differs


# ------------------------------
# Dummy Model & Config Fixtures
# ------------------------------


class DummyNERConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id2label = {0: "O", 1: "B-PROD", 2: "I-PROD"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = len(self.id2label)


class DummyNERModel(PreTrainedModel):
    config_class = DummyNERConfig

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(768, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        dummy_input = torch.randn(batch_size, seq_len, 768)
        logits = self.linear(dummy_input)
        return {"logits": logits}


@pytest.fixture
def dummy_model():
    config = DummyNERConfig()
    return DummyNERModel(config)


@pytest.fixture
def dummy_tokenizer():
    return MagicMock(spec=PreTrainedTokenizerBase)


@pytest.fixture
def dummy_dataset():
    return Dataset.from_dict(
        {"input_ids": [[1, 2, 3], [4, 5, 6]], "labels": [[0, 1, 2], [0, 1, 2]]}
    )


# ------------------------------
# NERTrainer Lifecycle Test
# ------------------------------


@patch("train.trainer.mlflow")
@patch("train.trainer.Trainer.train")
@patch("train.trainer.Trainer.evaluate")
@patch("train.trainer.Trainer.save_model")
@patch("train.trainer.DataCollatorForTokenClassification")
def test_ner_trainer_lifecycle(
    mock_data_collator,
    mock_save_model,
    mock_evaluate,
    mock_train,
    mock_mlflow,
    dummy_model,
    dummy_tokenizer,
    dummy_dataset,
):
    mock_train.return_value.metrics = {"train_loss": 0.1, "epoch": 3}
    mock_evaluate.return_value = {"eval_loss": 0.05, "f1": 0.88}
    mock_mlflow.set_experiment.return_value = None
    mock_mlflow.start_run.return_value.__enter__.return_value = None

    output_dir = Path("test_models")
    if output_dir.exists():
        for f in output_dir.iterdir():
            f.unlink()
        output_dir.rmdir()

    trainer = NERTrainer(
        model=dummy_model,
        tokenizer=dummy_tokenizer,
        train_dataset=dummy_dataset,
        eval_dataset=dummy_dataset,
        output_dir=output_dir,
        training_args_dict={"num_train_epochs": 1, "per_device_train_batch_size": 2},
        mlflow_experiment_name="test_experiment",
    )

    metrics = trainer.train()
    mock_train.assert_called_once()
    mock_mlflow.set_experiment.assert_called_once_with("NER_Training")
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_param.assert_any_call("epochs", 1)
    mock_mlflow.log_param.assert_any_call("train_batch_size", 2)
    mock_mlflow.log_metrics.assert_called_once_with(metrics)
    mock_save_model.assert_called_once()
    assert metrics == {"train_loss": 0.1, "epoch": 3}

    eval_metrics = trainer.evaluate()
    mock_evaluate.assert_called_once()
    assert eval_metrics == {"eval_loss": 0.05, "f1": 0.88}

    trainer.save_model()
    mock_save_model.assert_called()
    mock_mlflow.log_artifacts.assert_called()


# ------------------------------
# Metrics Computation Test
# ------------------------------


def test_compute_metrics(dummy_model):
    trainer = NERTrainer(
        model=dummy_model,
        tokenizer=MagicMock(),
        train_dataset=Dataset.from_dict({}),
        eval_dataset=Dataset.from_dict({}),  # Needed to avoid eval_strategy error
    )

    predictions = np.array(
        [[[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]], [[0.3, 0.4, 0.3], [0.05, 0.9, 0.05]]]
    )
    labels = [[1, 2], [0, -100]]
    result = trainer.compute_metrics((predictions, labels))

    assert all(k in result for k in ("precision", "recall", "f1", "accuracy"))
    assert all(0.0 <= v <= 1.0 for v in result.values())
