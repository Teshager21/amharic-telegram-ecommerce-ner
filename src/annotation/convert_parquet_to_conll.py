# src/annotation/convert_parquet_to_conll.py
"""
Convert NER annotations in Parquet format back to CoNLL format.
Author: Teshager Admasu
Date: 2025-06-20
"""

import pandas as pd
import argparse

# from pathlib import Path


def convert_parquet_to_conll(parquet_path: str, output_path: str):
    df = pd.read_parquet(parquet_path)

    required_cols = {"sentence_id", "token", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing one of required columns: {required_cols}")

    with open(output_path, "w", encoding="utf-8") as f:
        for sentence_id, group in df.groupby("sentence_id"):
            for _, row in group.iterrows():
                f.write(f"{row['token']}\t{row['label']}\n")
            f.write("\n")  # Sentence separator

    print(f"✅ Converted and saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NER parquet to CoNLL format")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input .parquet file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output .conll file"
    )
    args = parser.parse_args()

    convert_parquet_to_conll(args.input, args.output)
