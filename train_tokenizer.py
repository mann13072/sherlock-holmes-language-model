"""
Lesson 3: train the BPE tokenizer on the book.

Run:
    python train_tokenizer.py --input data/book.txt
Output:
    artifacts/tokenizer.json
"""

import argparse
import json

from src.dataset import load_book, split_train_val
from src.tokenizer import BookTokenizer, train_bpe
from src.utils import load_config, setup_console


def show(tokenizer: BookTokenizer, text: str) -> None:
    print("Text:")
    print(f"  {text}")
    print("Tokens:")
    print(f"  {tokenizer.tokens(text)}")
    print("Token IDs:")
    print(f"  {tokenizer.encode(text)}")
    print()


def main() -> None:
    setup_console()
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=cfg["data"]["book_path"])
    args = parser.parse_args()

    text = load_book(args.input)
    train_text, val_text = split_train_val(text, cfg["data"]["train_fraction"])

    # Train on the training text only. The validation text stays unseen,
    # so it is a fair test for the tokenizer and for the model.
    save_path = cfg["tokenizer"]["path"]
    tokenizer = train_bpe(train_text, cfg["tokenizer"]["vocab_size"], save_path)
    print(f"Saved: {save_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()

    # The first merges that BPE learned (the most frequent pairs).
    with open(save_path, encoding="utf-8") as f:
        merges = json.load(f)["model"]["merges"]
    print("First 10 merges (Ġ = space):")
    for i, m in enumerate(merges[:10], 1):
        a, b = m if isinstance(m, list) else m.split(" ")
        print(f"  {i:2d}. {a!r} + {b!r} -> {a + b!r}")
    print(f"  ... {len(merges)} merges in total")
    print()

    show(tokenizer, "The capital of France is Paris.")
    show(tokenizer, "Sherlock Holmes opened the door.")
    show(tokenizer, "Halifax")  # occurs only in the validation text

    # Check 1: encode then decode must give back the exact text.
    for name, part in [("train", train_text), ("validation", val_text)]:
        ids = tokenizer.encode(part)
        same = tokenizer.decode(ids) == part
        print(f"{name:10s} tokens: {len(ids):,}   characters per token: {len(part) / len(ids):.2f}   "
              f"decode(encode(text)) == text: {same}")


if __name__ == "__main__":
    main()
