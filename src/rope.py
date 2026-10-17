"""
RoPE (Rotary Position Embedding): turn Q and K by an angle that depends on the position.

Token embeddings encode token identity. RoPE introduces sequence-position
information into attention.

    Q or K for one head [..., seq, 32]  ->  16 pairs (0,1), (2,3), ... (30,31)
    pair i at position p turns by the angle  p * theta_i,   theta_i = 10000^(-2i/32)

A turn changes the direction of each pair. It does not change the length.
After the turn, the score Q . K depends on the distance between two tokens.

RoPE has no learned parameters. The angles come from a fixed formula.
RoPE acts on Q and K only. It does not change E, V, or the MLP.
"""

import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, head_dimension: int, base: float = 10000.0):
        super().__init__()
        # A turn needs two numbers, so the head size must be even.
        assert head_dimension % 2 == 0, "head_dimension must be even (RoPE uses pairs)"
        # One speed for each pair: theta_i = base^(-2i / head_dimension).
        # Pair 0 turns 1 rad per position. Pair 15 turns 0.000178 rad per position.
        i = torch.arange(head_dimension // 2, dtype=torch.float32)
        theta = base ** (-2 * i / head_dimension)
        # A buffer moves with the model to a device, but it is not a parameter.
        # The optimizer does not change it.
        self.register_buffer("theta", theta, persistent=False)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        # x: [..., seq, head_dimension], for example [batch, heads, seq, 32].
        # position_offset: the position of the first token in x.
        # The KV cache (Lesson 10) gives one new token at a time, at position offset.
        seq = x.shape[-2]
        positions = torch.arange(position_offset, position_offset + seq,
                                 dtype=self.theta.dtype, device=x.device)
        angles = positions[:, None] * self.theta[None, :]       # [seq, 16]
        cos, sin = angles.cos(), angles.sin()

        # Adjacent pairs: x0 = numbers 0, 2, 4, ...  x1 = numbers 1, 3, 5, ...
        x0, x1 = x[..., 0::2], x[..., 1::2]                      # [..., seq, 16] each
        # The 2x2 turn R(a) = [[cos a, -sin a], [sin a, cos a]] for each pair.
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        # Put each pair back in its place: (y0[0], y1[0], y0[1], y1[1], ...).
        return torch.stack((y0, y1), dim=-1).flatten(-2)


if __name__ == "__main__":
    # Lesson 5b check: run with  python -m src.rope
    import math

    from src.utils import load_config, set_seed, setup_console

    setup_console()
    cfg = load_config()
    set_seed(cfg["seed"])
    m = cfg["model"]
    H, T, DH = m["num_attention_heads"], m["context_length"], m["head_dimension"]

    rope = RoPE(DH)
    print(f"Learned parameters: {sum(p.numel() for p in rope.parameters())}")
    print(f"theta (16 pairs):   p0 = {rope.theta[0]:.4f}, p1 = {rope.theta[1]:.4f}, "
          f"..., p15 = {rope.theta[-1]:.6f}  rad per position")
    print()

    # 1. The 2x2 turn: one pair, by hand compared with the code.
    q = torch.tensor([[0.3026, 0.1034]]).repeat(3, 1)            # 3 positions, 1 pair
    alpha = 2 * 1.0                                              # position 2, theta_0 = 1 rad
    R = torch.tensor([[math.cos(alpha), -math.sin(alpha)],
                      [math.sin(alpha), math.cos(alpha)]])
    by_hand = R @ q[2]
    by_code = RoPE(2)(q)[2]
    print("1. One pair (0.3026, 0.1034) at position 2, turned by 2 rad:")
    print(f"   by hand R(2) @ v: ({by_hand[0]:+.4f}, {by_hand[1]:+.4f})")
    print(f"   by src/rope.py:   ({by_code[0]:+.4f}, {by_code[1]:+.4f})   "
          f"same: {torch.allclose(by_hand, by_code, atol=1e-6)}")
    print()

    # 2. The length does not change (the official check).
    Q = torch.randn(1, H, T, DH)                                 # [batch, heads, 256 positions, 32]
    Qr = rope(Q)
    change = (Qr.norm(dim=-1) - Q.norm(dim=-1)).abs().max().item()
    pair_change = (Qr.view(1, H, T, DH // 2, 2).norm(dim=-1)
                   - Q.view(1, H, T, DH // 2, 2).norm(dim=-1)).abs().max().item()
    print(f"2. Length, {H} heads x {T} positions:")
    print(f"   largest change of a vector length: {change:.1e}")
    print(f"   largest change of a pair length:   {pair_change:.1e}")
    print(f"   position 0 does not turn:          {torch.equal(Qr[..., 0, :], Q[..., 0, :])}")
    print()

    # 3. The score depends on the distance m - n, not on m and n.
    q1, k1 = torch.randn(DH), torch.randn(DH)

    def score(mq: int, nk: int) -> float:
        qm = rope(q1.expand(mq + 1, DH))[mq]                     # q1 at position mq
        kn = rope(k1.expand(nk + 1, DH))[nk]                     # k1 at position nk
        return (qm @ kn).item()

    print(f"3. Score of the same q and k (no RoPE: {(q1 @ k1).item():+.4f} at every place):")
    print("      m     n   m-n    score")
    for mq, nk in [(5, 2), (105, 102), (250, 247), (5, 4), (200, 2)]:
        print(f"   {mq:4d}  {nk:4d}  {mq - nk:4d}  {score(mq, nk):+.4f}")
    same = math.isclose(score(5, 2), score(250, 247), abs_tol=1e-4)
    print(f"   same distance -> same score: {same}")
    print()

    # 4. position_offset (for the KV cache in Lesson 10).
    one = rope(Q[..., 5:6, :], position_offset=5)
    print(f"4. One token with position_offset=5 == position 5 of the full piece: "
          f"{torch.allclose(one, Qr[..., 5:6, :], atol=1e-6)}")
