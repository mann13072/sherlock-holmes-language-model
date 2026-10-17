"""
Parameter inspector: where the learned numbers of Model A and Model B are.

A parameter is a learned number. The optimizer changes it in training.
W_Q, W_K, W_V, W_O, W_gate, W_up, W_down, the RMSNorm gains, E, and P are parameters.
RoPE angles and the frozen Gemma table are buffers: fixed numbers, not learned.

Run from the project root:  python inspect_parameters.py
"""

from src.model import build_model
from src.utils import load_config, setup_console


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def inspect(cfg: dict, which: str) -> None:
    model, tokenizer = build_model(cfg, which)
    emb = model.embedding
    blocks = model.blocks
    print(f"===== Model {which}  (vocabulary {tokenizer.vocab_size:,}) =====")

    if which == "A":
        print(f"Token embedding E:            {list(emb.weight.shape)}  = {count(emb):>10,}")
    else:
        print(f"Gemma table (buffer, frozen): {list(emb.table.shape)}  = {emb.table.numel():>10,}  (not learned)")
        print(f"Projection P:                 {list(emb.P.weight.shape)}  = {count(emb):>10,}")

    attn = [count(b.attention) for b in blocks]
    mlp = [count(b.mlp) for b in blocks]
    norm = sum(count(b.norm1) + count(b.norm2) for b in blocks) + count(model.final_norm)
    print(f"Attention per layer:          4 x {list(blocks[0].attention.W_Q.weight.shape)}  = {attn[0]:>10,}")
    print(f"MLP per layer:                3 x {list(blocks[0].mlp.W_up.weight.shape)} = {mlp[0]:>10,}")
    print(f"Normalization (all):          {2 * len(blocks) + 1} x {list(model.final_norm.gain.shape)}       = {norm:>10,}")
    print(f"Output layer:                 uses the embedding   = {0:>10,} new")

    total = count(model)
    groups = count(emb) + sum(attn) + sum(mlp) + norm
    print(f"Total parameters:                                {total:>10,}")
    assert groups == total, "the groups must add up to the total (the tied table counts one time)"

    a = blocks[0].attention
    print("Layer 0:")
    for name in ("W_Q", "W_K", "W_V", "W_O"):
        w = getattr(a, name).weight
        print(f"  {name} shape {list(w.shape)}   learned: {w.requires_grad}   first numbers: "
              f"{[round(v, 4) for v in w[0, :3].tolist()]}")
    print()


if __name__ == "__main__":
    setup_console()
    cfg = load_config()
    for which in ("A", "B"):
        inspect(cfg, which)
