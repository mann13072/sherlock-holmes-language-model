"""
Data pipeline, part 1: turn the raw book file into clean training text.

    data/book.txt  ->  load + normalize  ->  sequential 90/10 split
                                             (train text, validation text)

Data pipeline, part 2: turn the text into token IDs and cut training examples.

    text -> tokenizer -> one long list of IDs -> random windows
                                                 input  = ids[i     : i + T]
                                                 target = ids[i + 1 : i + T + 1]
"""

import re

import torch

# Project Gutenberg wraps every book in licence text. We cut it away,
# otherwise the model would also learn to write licence boilerplate.
GUTENBERG_START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*")
GUTENBERG_END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG")

# "Unusual" whitespace: tabs, non-breaking spaces, thin spaces, zero-width
# characters, etc. All become a plain space.
UNUSUAL_WHITESPACE = re.compile(r"[\t  -​  　﻿]")


def strip_gutenberg_boilerplate(text: str) -> str:
    """Keep only the text between the START and END markers, if present."""
    start = GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = GUTENBERG_END.search(text)
    if end:
        text = text[:end.start()]
    return text


def normalize_text(text: str) -> str:
    """Clean whitespace while preserving paragraph breaks and all punctuation.

    Gutenberg text is hard-wrapped at ~70 characters, so a single newline is
    just a line wrap, not a real break. Blank lines mark real paragraphs.

        "He was, I take it, the most\nperfect reasoning"   (line wrap)
     -> "He was, I take it, the most perfect reasoning"

    Paragraphs are re-joined with a blank line ("\n\n"), so the model can still
    learn where paragraphs (and dialogue turns) begin.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = UNUSUAL_WHITESPACE.sub(" ", text)

    paragraphs = re.split(r"\n\s*\n", text)        # blank line = paragraph break
    cleaned = []
    for p in paragraphs:
        p = p.replace("\n", " ")                   # undo hard line-wraps
        p = re.sub(r" {2,}", " ", p).strip()       # collapse repeated spaces
        if p:
            cleaned.append(p)
    return "\n\n".join(cleaned)


def load_book(path: str) -> str:
    """Read the book as UTF-8 and return normalized text.

    'utf-8-sig' silently drops a byte-order mark if the file starts with one.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    return normalize_text(strip_gutenberg_boilerplate(raw))


def split_train_val(text: str, train_fraction: float = 0.9) -> tuple[str, str]:
    """Split the book sequentially: the first part trains, the last part validates.

    We do NOT shuffle. Shuffling sentences would leak near-identical context
    from validation into training and make validation loss look better than it is.
    The validation text is never used to update parameters, only to measure how
    well the model handles text it has not trained on.

    The cut is moved forward to the next paragraph break so no sentence is split.
    """
    cut = int(len(text) * train_fraction)
    next_break = text.find("\n\n", cut)
    if next_break != -1:
        cut = next_break
    return text[:cut], text[cut:].lstrip("\n")


def load_token_ids(cfg: dict, tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the train text and the validation text one time.

    The result is two long 1-D tensors of token IDs. All training examples
    are cut from these tensors.
    """
    text = load_book(cfg["data"]["book_path"])
    train_text, val_text = split_train_val(text, cfg["data"]["train_fraction"])
    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    return train_ids, val_ids


def get_batch(ids: torch.Tensor, batch_size: int, context_length: int,
              device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Cut `batch_size` random windows from `ids`. Return (input, target).

    Each window has context_length + 1 tokens:
        input  = the first context_length tokens
        target = the last context_length tokens (the input, shifted by one)

    So target[b, t] is the token that comes after input[b, t].
    One window gives context_length predictions at the same time.

    Shapes: input [batch_size, context_length], target [batch_size, context_length]
    """
    # The last valid start is len - context_length - 1, so that the target fits.
    starts = torch.randint(len(ids) - context_length, (batch_size,))
    x = torch.stack([ids[i : i + context_length] for i in starts])
    y = torch.stack([ids[i + 1 : i + context_length + 1] for i in starts])
    return x.to(device), y.to(device)


def lesson2_check(cfg: dict) -> None:
    path = cfg["data"]["book_path"]

    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    text = load_book(path)
    train_text, val_text = split_train_val(text, cfg["data"]["train_fraction"])

    print(f"Raw file characters:      {len(raw):,}")
    print(f"Clean book characters:    {len(text):,}")
    print(f"Words (approx):           {len(text.split()):,}")
    print(f"Paragraphs:               {text.count(chr(10) * 2) + 1:,}")
    print(f"Distinct characters:      {len(set(text))}")
    print()
    print(f"Train characters:         {len(train_text):,}  ({len(train_text) / len(text):.1%})")
    print(f"Validation characters:    {len(val_text):,}  ({len(val_text) / len(text):.1%})")
    print(f"Lost at the seam:         {len(text) - len(train_text) - len(val_text)} (just the paragraph break)")
    print(f"Licence text removed:     {'Project Gutenberg' not in text}")
    print()

    paragraphs = text.split("\n\n")
    story_start = next(i for i, p in enumerate(paragraphs) if p.startswith("To Sherlock Holmes"))
    print("--- First story paragraph (line-wraps undone) ---")
    print(paragraphs[story_start][:400], "...")
    print()
    print("--- Last 200 chars of TRAIN ---")
    print(repr(train_text[-200:]))
    print()
    print("--- First 200 chars of VALIDATION ---")
    print(repr(val_text[:200]))


def lesson4_check(cfg: dict) -> None:
    from src.tokenizer import BookTokenizer
    from src.utils import set_seed

    set_seed(cfg["seed"])
    tokenizer = BookTokenizer.load(cfg["tokenizer"]["path"])
    train_ids, val_ids = load_token_ids(cfg, tokenizer)
    B = cfg["training"]["batch_size"]
    T = cfg["model"]["context_length"]

    print(f"Train IDs:       {len(train_ids):,}   first 10: {train_ids[:10].tolist()}")
    print(f"Validation IDs:  {len(val_ids):,}")
    print(f"Possible window starts in train: {len(train_ids) - T:,}")
    print()

    x, y = get_batch(train_ids, B, T)
    print(f"Input shape:  {list(x.shape)}   (batch_size, context_length)")
    print(f"Target shape: {list(y.shape)}")
    print(f"Predictions per batch: {B} x {T} = {B * T:,}")
    print(f"Steps to see each train token about one time: {len(train_ids) / (B * T):.0f}")
    print()

    # The check: the target is the input shifted by one token.
    shifted = torch.equal(y[:, :-1], x[:, 1:])
    print(f"target[:, :-1] == input[:, 1:]  ->  {shifted}")
    print()

    print("Example 0, first 8 positions:")
    print(f"  {'pos':>3}  {'input':>14}  ->  {'target':<14}")
    for t in range(8):
        a = tokenizer.decode([x[0, t].item()])
        b = tokenizer.decode([y[0, t].item()])
        print(f"  {t:>3}  {a!r:>14}  ->  {b!r:<14}")


if __name__ == "__main__":
    # Run with:  python -m src.dataset            (Lesson 4 check)
    #            python -m src.dataset --lesson 2 (Lesson 2 check)
    import argparse

    from src.utils import load_config, setup_console

    setup_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lesson", type=int, default=4, choices=[2, 4])
    config = load_config()
    if parser.parse_args().lesson == 2:
        lesson2_check(config)
    else:
        lesson4_check(config)
