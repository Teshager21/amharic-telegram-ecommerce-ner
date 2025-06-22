# scripts/split_conll.py

import random

# from pathlib import Path


def split_conll_file(input_path, train_path, eval_path, eval_ratio=0.1, seed=42):
    """
    Splits a .conll file into train and eval files by sentences.

    Args:
        input_path (str or Path): Path to the original .conll file.
        train_path (str or Path): Path to output train .conll file.
        eval_path (str or Path): Path to output eval .conll file.
        eval_ratio (float): Fraction of sentences to use for eval.
        seed (int): Random seed for reproducibility.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    # Group lines into sentences (split by empty line)
    sentences = []
    current_sentence = []
    for line in lines:
        if line.strip() == "":
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            current_sentence.append(line)
    # Add last sentence if no trailing newline
    if current_sentence:
        sentences.append(current_sentence)

    random.seed(seed)
    random.shuffle(sentences)

    split_idx = int(len(sentences) * (1 - eval_ratio))
    train_sentences = sentences[:split_idx]
    eval_sentences = sentences[split_idx:]

    def write_sentences(sentences_list, filepath):
        with open(filepath, "w", encoding="utf-8") as f_out:
            for sent in sentences_list:
                f_out.write("\n".join(sent))
                f_out.write("\n\n")  # empty line between sentences

    write_sentences(train_sentences, train_path)
    write_sentences(eval_sentences, eval_path)

    print(f"Split {len(sentences)} sentences into:")
    print(f" - Train: {len(train_sentences)} sentences -> {train_path}")
    print(f" - Eval: {len(eval_sentences)} sentences -> {eval_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split .conll file into train/eval")
    parser.add_argument(
        "--input", type=str, required=True, help="Original .conll file path"
    )
    parser.add_argument(
        "--train_output", type=str, required=True, help="Output train .conll file path"
    )
    parser.add_argument(
        "--eval_output", type=str, required=True, help="Output eval .conll file path"
    )
    parser.add_argument(
        "--eval_ratio", type=float, default=0.1, help="Eval split ratio (default=0.1)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    split_conll_file(
        args.input, args.train_output, args.eval_output, args.eval_ratio, args.seed
    )
