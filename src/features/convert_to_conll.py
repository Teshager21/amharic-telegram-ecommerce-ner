# src/features/convert_to_conll.py
"""
Convert labeled token/tag dataset to CoNLL-format .txt file for NER training.

Author: Teshager Admasu
Date: 2025-06-19
"""

import logging
from pathlib import Path

# from typing import List, Tuple
import pandas as pd


logging.basicConfig(level=logging.INFO, format="📦 [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def dataframe_to_conll(
    df: pd.DataFrame,
    token_col: str,
    tag_col: str,
    message_id_col: str,
    output_path: Path,
) -> None:
    """
    Convert labeled DataFrame to CoNLL format and save as .txt.

    Args:
        df (pd.DataFrame): DataFrame containing labeled tokens and tags.
        token_col (str): Column name for tokens.
        tag_col (str): Column name for NER tags.
        message_id_col (str): Column name indicating message/sentence grouping.
        output_path (Path): Output file path for CoNLL .txt file.

    Raises:
        ValueError: If required columns are missing.
    """
    required_cols = {token_col, tag_col, message_id_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Converting DataFrame to CoNLL format at {output_path}")

    with output_path.open("w", encoding="utf-8") as f:
        current_msg_id = None
        for _, row in df.iterrows():
            msg_id = row[message_id_col]

            # Blank line between messages
            if msg_id != current_msg_id:
                if current_msg_id is not None:
                    f.write("\n")
                current_msg_id = msg_id

            token = str(row[token_col])
            tag = str(row[tag_col])
            f.write(f"{token} {tag}\n")

    logger.info(f"Saved CoNLL-formatted file: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert labeled data to CoNLL .txt format"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input CSV file path with labeled tokens and tags",
    )
    parser.add_argument(
        "--token-col",
        type=str,
        default="token",
        help="Column name for tokens (default: token)",
    )
    parser.add_argument(
        "--tag-col",
        type=str,
        default="tag",
        help="Column name for NER tags (default: tag)",
    )
    parser.add_argument(
        "--msg-id-col",
        type=str,
        default="message_id",
        help="Column name to group tokens into messages/sentences"
        " (default: message_id)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output file path for CoNLL .txt file",
    )

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input_csv)
        dataframe_to_conll(
            df,
            token_col=args.token_col,
            tag_col=args.tag_col,
            message_id_col=args.msg_id_col,
            output_path=args.output_path,
        )
    except Exception as e:
        logger.error(f"Failed to convert to CoNLL format: {e}")


if __name__ == "__main__":
    main()
