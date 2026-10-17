"""
LoRA (low-rank adaptation): fine-tune a trained model without a change to its weights.

Each chosen nn.Linear W [out, in] stays frozen. Two small matrices learn beside it:
    A [r, in]   (random start)
    B [out, r]  (zero start, so the model at step 0 is exactly the trained model)
The layer computes  W·h + (alpha / r)·B·A·h.  B·A has rank r at most: a small change, in a few directions only.

At the end, merged_state_dict() adds (alpha / r)·B·A into each W. The result has the same keys as a model without
LoRA, so every script can load it.
"""

import math

import torch
import torch.nn as nn

TARGETS = ("W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))          # the start of the LoRA paper (as nn.Linear)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        self.scale = alpha / rank

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.base(h) + (h @ self.A.T @ self.B.T) * self.scale

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scale * (self.B @ self.A)


def add_lora(model: nn.Module, rank: int, alpha: float) -> int:
    """Freeze all parameters, then put a LoRALinear around each TARGETS layer. Return the number of trainable numbers."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name in TARGETS:
            child = getattr(module, name, None)
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def has_lora(model: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in model.modules())


@torch.no_grad()
def merged_state_dict(model: nn.Module) -> dict:
    """The state dict of the same model without LoRA: each W becomes W + (alpha / r)·B·A."""
    lora = {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}
    out = {k: v for k, v in model.state_dict().items() if not any(k.startswith(n + ".") for n in lora)}
    for name, m in lora.items():
        out[name + ".weight"] = m.merged_weight().detach().clone()
    return out


if __name__ == "__main__":
    # Check: (1) at the start the output equals the trained model; (2) after a change to B, the merged plain model
    # gives the same output as the LoRA model; (3) only A and B are trainable.
    import sys
    sys.path.insert(0, ".")
    from src.model import build_model
    from src.utils import load_config, setup_console
    setup_console()
    cfg = load_config()
    ckpt = torch.load("artifacts/checkpoints/model_a_s42_d02_lr6e-4_e20/best.pt", map_location="cpu", weights_only=False)
    model, _ = build_model(ckpt["config"], "A")
    model.load_state_dict(ckpt["model"])
    model.eval()
    x = torch.randint(0, 4000, (2, 40))
    before = model(x)[0]
    n = add_lora(model, rank=8, alpha=16)
    model.eval()
    assert torch.allclose(model(x)[0], before, atol=1e-5), "B = 0 must give the trained model"
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.B.normal_(0, 0.01)
    lora_out = model(x)[0]
    plain, _ = build_model(ckpt["config"], "A")
    plain.load_state_dict(merged_state_dict(model))
    plain.eval()
    assert torch.allclose(plain(x)[0], lora_out, atol=1e-4), "the merged model must equal the LoRA model"
    assert all(k.endswith((".A", ".B")) for k, p in model.named_parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"checks passed: start = trained model; merged = LoRA model; trainable {n:,} of {total:,} ({n / total:.1%})")
