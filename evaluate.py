"""
Lesson 8: Evaluate Model A and Model B after training.

1. Load best.pt of each model (the step with the lowest validation loss).
2. Calculate the loss on the whole validation text: per token, per character, bits per character, perplexity.
3. Plot the train loss and the validation loss per character for both models: artifacts/plots/loss_curves.png.

The two tokenizers make different tokens, so the loss per token is not a fair comparison.
The loss per character is fair: both models read the same characters.

Run from the project root:  python evaluate.py
"""

import csv
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.dataset import load_book, split_train_val
from src.model import build_model
from src.utils import load_config, setup_console
from train import BookStream, full_loss

COLORS = {"A": "#2a78d6", "B": "#eb6834",     # categorical slot 1 and slot 2
          "C": "#1baf7a", "C random": "#eda100",   # slot 3 and slot 4: Model C (Gemma table) and its random-table control
          "C ctx256": "#e87ba4",                   # slot 5: Model C with context 256 (needs the direct "best" labels)
          "C256 + new data": "#008300", "C256 control": "#6250d6"}   # slot 6 and 7: the new-data run and its control


STEP_LINE = re.compile(r"^step\s+(\d+) \| train [\d.]+ \(([\d.]+)/char\) \| val [\d.]+ \(([\d.]+)/char")


def read_log(which: str) -> dict:
    """The evaluation rows of a run: from artifacts/logs/train_output_<which>.txt (since 2026-10-16 there is no CSV),
    or from the old artifacts/logs/train_log_<which>.csv."""
    rows = {"step": [], "train": [], "val": []}
    txt = os.path.join("artifacts", "logs", f"train_output_{which}.txt")
    if os.path.exists(txt):
        with open(txt, encoding="utf-8") as f:
            for line in f:
                if m := STEP_LINE.match(line):
                    rows["step"].append(int(m[1]))
                    rows["train"].append(float(m[2]))
                    rows["val"].append(float(m[3]))
        return rows
    with open(os.path.join("artifacts", "logs", f"train_log_{which}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["val_loss_per_char"]:
                rows["step"].append(int(r["step"]))
                rows["train"].append(float(r["train_loss_per_char"]))
                rows["val"].append(float(r["val_loss_per_char"]))
    return rows


def plot(logs: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, (which, rows) in enumerate(logs.items()):
        c = COLORS[which]
        ax.plot(rows["step"], rows["train"], color=c, lw=2, ls="--", label=f"Model {which} train")
        ax.plot(rows["step"], rows["val"], color=c, lw=2, label=f"Model {which} validation")
        best = min(range(len(rows["val"])), key=rows["val"].__getitem__)
        ax.plot(rows["step"][best], rows["val"][best], "o", ms=8, color=c, mec="#fcfcfb", mew=2)
        ax.annotate(f"{which} best {rows['val'][best]:.3f}\n(step {rows['step'][best]})",
                    (rows["step"][best], rows["val"][best]), textcoords="offset points", xytext=(6, 10 if i % 2 == 0 else -32),   # alternate: no overlap
                    fontsize=9, color="#52514e")
    ax.set_xlabel("training step", color="#52514e")
    ax.set_ylabel("loss per character (nats)", color="#52514e")
    ax.set_title("Train and validation loss per character", loc="left", color="#0b0b0b", fontsize=13)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, fontsize=9, labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot saved: {path}")


def main() -> None:
    setup_console()
    cfg = load_config()
    T, B = cfg["model"]["context_length"], cfg["training"]["batch_size"]
    _, val_text = split_train_val(load_book(cfg["data"]["book_path"]), cfg["data"]["train_fraction"])

    print(f"{'':8} {'step':>5} {'loss/token':>11} {'perplexity':>11} {'loss/char':>10} {'bits/char':>10}")
    for which in ("A", "B"):
        model, tokenizer = build_model(cfg, which)
        ckpt = torch.load(os.path.join(cfg["training"]["checkpoint_dir"], f"model_{which.lower()}", "best.pt"),
                          weights_only=False)
        model.load_state_dict(ckpt["model"])
        per_token, per_char = full_loss(model, BookStream(tokenizer, val_text), T, B)
        assert abs(per_token - ckpt["val_loss"]) < 1e-4, "the checkpoint must give the same loss again"
        print(f"Model {which}  {ckpt['step']:>5} {per_token:>11.3f} {math.exp(per_token):>11.1f} "
              f"{per_char:>10.3f} {per_char / math.log(2):>10.3f}")

    plot({w: read_log(w) for w in ("A", "B")}, os.path.join("artifacts", "plots", "loss_curves.png"))


if __name__ == "__main__":
    main()
