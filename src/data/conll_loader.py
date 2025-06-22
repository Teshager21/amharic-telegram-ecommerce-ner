# src/data/conll_loader.py
"""
Module to load CoNLL-formatted NER datasets.
Author: Teshager Admasu
Date: 2025-06-19
"""

import logging
from pathlib import Path
import pandas as pd
from typing import Union

logger = logging.getLogger(__name__)


def load_conll_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CoNLL-formatted .txt file into a DataFrame.

    Args:
        file_path (Union[str, Path]): Path to the CoNLL .txt file.

    Returns:
        pd.DataFrame: DataFrame with columns [sentence_id, token, label]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"❌ File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    data = []
    sentence_id = 0

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                sentence_id += 1
                continue
            try:
                token, label = line.split()
                data.append(
                    {"sentence_id": sentence_id, "token": token, "label": label}
                )
            except ValueError:
                logger.warning(f"⚠️ Malformed line skipped: {line}")

    df = pd.DataFrame(data)
    logger.info(f"✅ Loaded {len(df)} tokens from {file_path}")
    return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="🔍 [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="📥 Load CoNLL-formatted data")
    parser.add_argument(
        "--file", type=str, required=True, help="Path to .txt CoNLL file"
    )
    args = parser.parse_args()

    df = load_conll_data(args.file)
    print(df.head())
