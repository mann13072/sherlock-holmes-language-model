"""
Shared helpers: config loading, seeding, device selection, console setup.

Nothing here is "AI" yet. These are the boring but necessary pieces every
other script imports.
"""

import random
import sys

import numpy as np
import torch
import yaml


def setup_console() -> None:
    """Make printing Unicode (arrows, book text, BPE symbols) safe on Windows.

    The Windows console often defaults to cp1252, which crashes on characters
    like '↓' or 'Ġ'. Switching stdout to UTF-8 avoids that.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_config(path: str = "config.yaml") -> dict:
    """Read config.yaml and attach the chosen model preset as cfg['model']."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    preset = cfg.get("preset", "default")
    cfg["model"] = dict(cfg["model_presets"][preset])
    cfg["model"]["preset"] = preset

    # The core relationship of multi-head attention:
    # the embedding vector is split evenly across the heads.
    m = cfg["model"]
    assert m["embedding_dimension"] == m["num_attention_heads"] * m["head_dimension"], (
        "embedding_dimension must equal num_attention_heads x head_dimension"
    )
    return cfg


def set_seed(seed: int) -> None:
    """Seed every random number generator we use, so runs are reproducible.

    Randomness appears in: weight initialization, batch sampling, dropout,
    and sampling during text generation.
    """
    random.seed(seed)        # Python
    np.random.seed(seed)     # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch (all GPUs)


def get_device() -> torch.device:
    """Pick the fastest available hardware: NVIDIA GPU -> Apple GPU -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    # Lesson 1 check: config loads, seeds work, device is detected.
    setup_console()
    cfg = load_config()
    set_seed(cfg["seed"])

    print("Config loaded. Preset:", cfg["model"]["preset"])
    for k, v in cfg["model"].items():
        print(f"  {k:24s} {v}")
    print("Device:", get_device())

    # Reproducibility demo: same seed -> same "random" numbers.
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    print("Random draw #1:", a.tolist())
    print("Random draw #2:", b.tolist())
    print("Identical after re-seeding:", torch.equal(a, b))
