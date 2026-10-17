"""
Token embeddings: token ID -> learned vector.

    token IDs [batch, seq]  ->  E[token IDs]  ->  vectors [batch, seq, embedding_dimension]

E is a table with one row for each token in the vocabulary.
The token ID selects one row. There is no other calculation.

Each dimension is a learned coordinate. Individual dimensions do not
normally have a fixed human-readable meaning. Meaning is spread across
many dimensions.

Token embeddings encode token identity. RoPE introduces sequence-position
information into attention (src/rope.py). The embedding has no position.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dimension: int):
        super().__init__()
        # The embedding matrix E, shape [vocab_size, embedding_dimension].
        # It is a parameter, so loss.backward() calculates a gradient for it
        # and the optimizer changes it.
        self.weight = nn.Parameter(torch.empty(vocab_size, embedding_dimension))

        # Random start: small values from a normal distribution (mean 0, std 0.02).
        # Random initialization breaks symmetry so different dimensions and heads
        # can learn different behaviors. If all rows were equal, all tokens would
        # get equal gradients and stay equal.
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup: each ID selects its row of E.
        # [batch, seq] -> [batch, seq, embedding_dimension]
        return self.weight[token_ids]

    def output_table(self) -> torch.Tensor:
        """E itself, [vocab_size, embedding_dimension]. The output layer uses it (weight tying)."""
        return self.weight


if __name__ == "__main__":
    # Lesson 5 (part 1) check: run with  python -m src.embeddings
    import torch.nn.functional as F

    from src.tokenizer import BookTokenizer
    from src.utils import load_config, set_seed, setup_console

    setup_console()
    cfg = load_config()
    set_seed(cfg["seed"])
    tokenizer = BookTokenizer.load(cfg["tokenizer"]["path"])
    V, D = tokenizer.vocab_size, cfg["model"]["embedding_dimension"]

    emb = TokenEmbedding(V, D)
    E = emb.weight
    print(f"E shape:          {list(E.shape)}")
    print(f"E parameters:     {E.numel():,}")
    print(f"E mean / std:     {E.mean().item():+.5f} / {E.std().item():.5f}   (target: 0 / 0.02)")
    print()

    sentence = "To Sherlock Holmes she is always _the_ woman."
    ids = torch.tensor([tokenizer.encode(sentence)])
    x = emb(ids)
    print(f"Sentence:         {sentence}")
    print(f"Token IDs shape:  {list(ids.shape)}")
    print(f"Vectors shape:    {list(x.shape)}")
    print()

    k = tokenizer.tokens(sentence).index(" Holmes")
    hid = ids[0, k].item()
    print(f"' Holmes' = ID {hid}")
    print(f"  x[0, {k}][:4]   = {[round(v, 4) for v in x[0, k, :4].tolist()]}")
    print(f"  E[{hid}][:4]  = {[round(v, 4) for v in E[hid, :4].tolist()]}")
    print(f"  lookup == row of E:     {torch.equal(x[0, k], E[hid])}")
    one_hot = F.one_hot(torch.tensor(hid), V).float()
    print(f"  one-hot @ E == lookup:  {torch.allclose(one_hot @ E, x[0, k])}")
    print()

    # Gradient check: only the rows of the tokens in the sentence get a gradient.
    x.sum().backward()
    rows = (E.grad.abs().sum(dim=1) > 0).nonzero().flatten().tolist()
    print(f"Rows with a gradient: {len(rows)} of {V:,}")
    print(f"Same as the sentence IDs: {sorted(set(ids[0].tolist())) == rows}")
