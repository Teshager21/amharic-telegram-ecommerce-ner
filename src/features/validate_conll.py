# src/features/validate_conll.py
"""
Validator for CoNLL-format .txt files for NER.

Author: Teshager Admasu
Date: 2025-06-19
"""

import logging
from pathlib import Path
import sys
from typing import List

logging.basicConfig(level=logging.INFO, format="📦 [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VALID_BIO_PREFIXES = {"B-", "I-", "O"}


def is_valid_bio_tag(tag: str) -> bool:
    if tag == "O":
        return True
    if any(tag.startswith(prefix) for prefix in ("B-", "I-")):
        return True
    return False


def validate_conll_file(file_path: Path) -> List[str]:
    """
    Validate a CoNLL file for well-formedness.

    Args:
        file_path (Path): Path to the CoNLL .txt file.

    Returns:
        List[str]: List of error messages found. Empty if no errors.
    """
    errors = []

    with file_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()

            if line == "":
                # Blank line separates sentences
                continue

            parts = line.split()
            if len(parts) != 2:
                errors.append(f"Line {lineno}: Expected 2 columns but got {len(parts)}")

            else:
                token, tag = parts
                if not is_valid_bio_tag(tag):
                    errors.append(f"Line {lineno}: Invalid BIO tag '{tag}'")

    return errors


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate CoNLL .txt file for NER format"
    )
    parser.add_argument(
        "file", type=Path, help="Path to CoNLL-format .txt file to validate"
    )

    args = parser.parse_args()

    logger.info(f"Validating file: {args.file}")

    errors = validate_conll_file(args.file)

    if errors:
        logger.error(f"Found {len(errors)} errors:")
        for err in errors:
            logger.error(f"  - {err}")
        sys.exit(1)

    logger.info("✅ No errors found. File is well-formed.")


if __name__ == "__main__":
    main()
