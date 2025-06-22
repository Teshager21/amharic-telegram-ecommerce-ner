import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from pathlib import Path

# Import the main entrypoint function to test
from src.train import main as train_entrypoint  # assuming main.py has a main() function


@pytest.fixture
def dummy_cfg(tmp_path):
    return OmegaConf.create(
        {
            "logging": {"level": "INFO"},
            "data": {
                "train_file": str(tmp_path / "dummy_train.conll"),
                "eval_file": str(tmp_path / "dummy_eval.conll"),
            },
            "model": {
                "name_or_path": "dummy-model",
                "max_length": 128,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 5e-5,
                "batch_size": 2,
                "seed": 42,
                "eval_strategy": "epoch",
                "label_all_tokens": True,
            },
            "output_dir": str(tmp_path / "model_output"),
        }
    )


@patch("src.train.main.AutoTokenizer.from_pretrained")
@patch("src.train.main.AutoModelForTokenClassification.from_pretrained")
@patch("src.train.main.load_conll_data")
@patch("src.train.main.prepare_tokenizer_and_align_labels")
@patch("src.train.main.NERTrainer")
def test_main_train_pipeline(
    mock_trainer,
    mock_align_labels,
    mock_load_data,
    mock_model_loader,
    mock_tokenizer_loader,
    dummy_cfg,
):
    import pandas as pd

    # Setup dummy dataframe for load_conll_data
    mock_df = pd.DataFrame(
        {
            "sentence_id": [0, 0, 1, 1],
            "token": ["Token1", "Token2", "Token3", "Token4"],
            "label": ["B-ENT", "I-ENT", "B-ENT", "I-ENT"],
        }
    )
    mock_load_data.return_value = mock_df

    # Setup return for prepare_tokenizer_and_align_labels
    mock_align_labels.return_value = (
        [[101, 102], [103, 104]],  # dummy input_ids
        [[0, 1], [0, 1]],  # dummy labels
    )

    # Dummy tokenizer and model mocks
    mock_tokenizer_loader.return_value = MagicMock()
    mock_model_loader.return_value = MagicMock()

    # Dummy trainer instance mock
    trainer_instance = MagicMock()
    mock_trainer.return_value = trainer_instance

    # Call the main function
    train_entrypoint.main(dummy_cfg)

    # Assertions
    from unittest.mock import call

    mock_load_data.assert_has_calls(
        [call(Path(dummy_cfg.data.train_file)), call(Path(dummy_cfg.data.eval_file))]
    )
    assert mock_load_data.call_count == 2

    from unittest.mock import call

    expected_call = call(
        texts=[["Token1", "Token2"], ["Token3", "Token4"]],
        labels=[[0, 1], [0, 1]],
        tokenizer=mock_tokenizer_loader.return_value,
        label_all_tokens=True,
        max_length=128,
    )
    mock_align_labels.assert_has_calls([expected_call, expected_call])
    assert mock_align_labels.call_count == 2

    mock_tokenizer_loader.assert_called_once_with("dummy-model")
    mock_model_loader.assert_called_once()
    trainer_instance.train.assert_called_once()
    trainer_instance.evaluate.assert_called_once()
    trainer_instance.save_model.assert_called_once()
