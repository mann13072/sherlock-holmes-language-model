"""
Causal multi-head self-attention, calculated step by step.

    h [batch, seq, 256]
      -> Q = h W_Q,  K = h W_K,  V = h W_V          each [batch, seq, 256]
      -> cut into 8 heads                            each [batch, 8, seq, 32]
      -> RoPE turns Q and K (not V)
      -> scores = Q K^T / sqrt(32)                   [batch, 8, seq, seq]
      -> causal mask: future scores = -inf
      -> softmax: each row becomes weights that add up to 1
      -> weights x V                                 [batch, 8, seq, 32]
      -> join the 8 heads, then W_O                  [batch, seq, 256]

Q: what information a token looks for.  K: what a token offers for the comparison.
V: the information that attention collects.

The calculation does not use F.scaled_dot_product_attention. Each step is
explicit, so the inspection tools (Lesson 11) can read Q, K, V, the scores,
and the weights. Set `inspect = True` to keep them in `self.saved`.

No head has a fixed job. Each head develops its own pattern during training.
"""

import math

import torch
import torch.nn as nn

from src.rope import RoPE


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dimension: int, num_heads: int, head_dimension: int, dropout: float = 0.0):
        super().__init__()
        assert embedding_dimension == num_heads * head_dimension, "embedding_dimension = heads x head_dimension"
        self.num_heads = num_heads
        self.head_dimension = head_dimension

        # Four learned matrices, 256 x 256 each, no bias.
        # nn.Linear stores the weight as [out, in] and calculates h @ weight.T.
        self.W_Q = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.W_K = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.W_V = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.W_O = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        # Random start (mean 0, std 0.02). Random values break symmetry,
        # so different heads can learn different behaviors.
        for layer in (self.W_Q, self.W_K, self.W_V, self.W_O):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

        self.rope = RoPE(head_dimension)
        self.attention_dropout = nn.Dropout(dropout)   # on the weights, during training only
        self.output_dropout = nn.Dropout(dropout)      # on the output, during training only

        self.inspect = False
        self.saved: dict[str, torch.Tensor] = {}

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # [batch, seq, 256] -> [batch, seq, 8, 32] -> [batch, 8, seq, 32]. No new numbers.
        batch, seq, _ = t.shape
        return t.view(batch, seq, self.num_heads, self.head_dimension).transpose(1, 2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = h.shape

        q = self.split_heads(self.W_Q(h))                              # [batch, 8, seq, 32]
        k = self.split_heads(self.W_K(h))
        v = self.split_heads(self.W_V(h))
        q, k = self.rope(q), self.rope(k)                              # position enters here

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dimension)   # [batch, 8, seq, seq]
        future = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=h.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))            # a token cannot see later tokens
        weights = torch.softmax(scores, dim=-1)                        # each row adds up to 1

        out = self.attention_dropout(weights) @ v                      # [batch, 8, seq, 32]
        out = out.transpose(1, 2).contiguous().view(batch, seq, dim)   # join the heads: [batch, seq, 256]
        out = self.output_dropout(self.W_O(out))

        if self.inspect:
            self.saved = {name: t.detach() for name, t in
                          dict(q=q, k=k, v=v, scores=scores, weights=weights).items()}
        return out


if __name__ == "__main__":
    # Lesson 6 check: run with  python -m src.attention
    from src.embeddings import TokenEmbedding
    from src.tokenizer import BookTokenizer
    from src.utils import load_config, set_seed, setup_console

    setup_console()
    cfg = load_config()
    set_seed(cfg["seed"])
    m = cfg["model"]
    tokenizer = BookTokenizer.load(cfg["tokenizer"]["path"])
    D, H, DH = m["embedding_dimension"], m["num_attention_heads"], m["head_dimension"]

    emb = TokenEmbedding(tokenizer.vocab_size, D)
    attn = CausalSelfAttention(D, H, DH, dropout=m["dropout"]).eval()   # eval: dropout off
    attn.inspect = True

    def rmsnorm(x):  # the planned formula; src/rmsnorm.py comes in Lesson 7
        return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    sentence = "To Sherlock Holmes she is always _the_ woman."
    ids = torch.tensor([tokenizer.encode(sentence)])
    names = tokenizer.tokens(sentence)
    T = ids.shape[1]
    with torch.no_grad():
        out = attn(rmsnorm(emb(ids)))
    s = attn.saved
    print(f"Parameters:  {sum(p.numel() for p in attn.parameters()):,}  (4 x {D} x {D})")
    print(f"h       {[1, T, D]}")
    print(f"Q, K, V {list(s['q'].shape)}   (batch, heads, tokens, numbers)")
    print(f"weights {list(s['weights'].shape)}")
    print(f"output  {list(out.shape)}")
    print()

    w = s["weights"][0]                                                 # [8, T, T]
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    print(f"Each row of weights adds up to 1:   {torch.allclose(w.sum(-1), torch.ones(H, T))}")
    print(f"Weight on a future token is 0:      {bool((w[:, future] == 0).all())}")
    k = names.index(" woman")
    row = w[0, k, :k + 1]
    print(f"Head 0, ' woman' (position {k}) looks at {k + 1} tokens:")
    print("   " + "  ".join(f"{n.strip() or n!r}:{v:.3f}" for n, v in zip(names, row.tolist())))
    print(f"   equal share would be 1/{k + 1} = {1 / (k + 1):.3f}  (before training)")
    print()

    # The main check: change one token. The outputs before it must not change.
    for pos, new in [(T - 1, "!"), (5, " never")]:
        ids2 = ids.clone()
        ids2[0, pos] = tokenizer.encode(new)[0]
        with torch.no_grad():
            out2 = attn(rmsnorm(emb(ids2)))
        change = (out2 - out).abs().amax(dim=-1)[0]
        print(f"Change position {pos} ('{names[pos]}' -> '{new}'). Largest change of the output, per position:")
        print("   " + " ".join(f"{c:.1e}" if c > 0 else "0" for c in change.tolist()))
        print(f"   positions 0..{pos - 1} unchanged: {bool((change[:pos] == 0).all())}"
              f"   |   positions {pos}..{T - 1} changed: {bool((change[pos:] > 0).all())}")
