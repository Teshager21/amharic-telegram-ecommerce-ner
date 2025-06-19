# src/annotation/convert_bio_to_parquet.py

"""
Convert BIO-formatted labeled data to a structured Parquet file.
Author: Teshager Admasu
Date: 2025-06-19
"""

import logging
import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="🔎 [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_bio_file(file_path: Path) -> pd.DataFrame:
    """
    Parse a raw BIO-tagged file into a DataFrame.

    Args:
        file_path (Path): Path to the .txt file with BIO-tagged tokens.

    Returns:
        pd.DataFrame: Structured DataFrame with sentence_id, token, label.
    """
    tokens: List[str] = []
    labels: List[str] = []
    sentence_ids: List[int] = []

    current_sentence: List[Tuple[str, str]] = []
    sentence_id = 0

    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:  # New sentence
                    if current_sentence:
                        for token, label in current_sentence:
                            tokens.append(token)
                            labels.append(label)
                            sentence_ids.append(sentence_id)
                        sentence_id += 1
                        current_sentence = []
                else:
                    try:
                        token, label = line.rsplit(" ", 1)
                        current_sentence.append((token, label))
                    except ValueError:
                        logger.warning(f"Skipping malformed line: '{line}'")

            # Final sentence flush
            if current_sentence:
                for token, label in current_sentence:
                    tokens.append(token)
                    labels.append(label)
                    sentence_ids.append(sentence_id)

        logger.info(f"✅ Parsed {sentence_id + 1} sentences from {file_path.name}")
        return pd.DataFrame(
            {"sentence_id": sentence_ids, "token": tokens, "label": labels}
        )

    except FileNotFoundError:
        logger.error(f"❌ File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error while parsing: {e}")
        raise


def save_to_parquet(df: pd.DataFrame, output_path: Path):
    """
    Save DataFrame to a Parquet file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_path (Path): Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"📦 Saved structured data to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BIO-tagged text file to Parquet format."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input .txt file with BIO-labeled data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/ner_annotations.parquet",
        help="Path to save the output Parquet file",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = parse_bio_file(input_path)
    save_to_parquet(df, output_path)


if __name__ == "__main__":
    main()
