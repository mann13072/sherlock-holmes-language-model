"""
The full model: token IDs -> logits -> loss.

    token IDs [batch, seq]
      -> embedding                     [batch, seq, 256]   Model A: E.  Model B: P . (Gemma row)
      -> 6 Transformer blocks          [batch, seq, 256]
      -> final RMSNorm                 [batch, seq, 256]
      -> h . (output table)^T          [batch, seq, vocab_size]   = logits
      -> cross-entropy with targets    one number = loss

Weight tying: the output layer has no matrix of its own. It uses the embedding table again.
Model A: logit of token i = E[i] . h.   Model B: logit of token i = (P . Gemma row i) . h.
Exception: Model C (model_c.untied_output) has a separate output head [4000, 256]: logit of token i = W_out[i] . h.

The model accepts either embedding module. Both have the same forward shapes and an output_table().
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rmsnorm import RMSNorm
from src.transformer_block import TransformerBlock


class BookTransformer(nn.Module):
    def __init__(self, embedding: nn.Module, m: dict, untied_vocab_size: int | None = None):
        super().__init__()
        self.embedding = embedding
        self.blocks = nn.ModuleList([
            TransformerBlock(m["embedding_dimension"], m["num_attention_heads"], m["head_dimension"],
                             m["feed_forward_dimension"], m["dropout"])
            for _ in range(m["num_transformer_layers"])])
        self.final_norm = RMSNorm(m["embedding_dimension"])
        # Untied output (Model C): a separate head [vocab_size, 256]. It is made after the blocks,
        # so the blocks start with the same numbers as in Model A. Start std 0.02, the same as E in Model A.
        self.output = None
        if untied_vocab_size:
            self.output = nn.Linear(m["embedding_dimension"], untied_vocab_size, bias=False)
            nn.init.normal_(self.output.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embedding(token_ids)                    # [batch, seq, 256]
        for block in self.blocks:
            x = block(x)                                 # [batch, seq, 256]
        h = self.final_norm(x)
        if self.output is not None:
            logits = self.output(h)                      # [batch, seq, vocab_size], untied head
        else:
            logits = h @ self.embedding.output_table().T     # [batch, seq, vocab_size]
        if targets is None:
            return logits, None
        # Cross-entropy: mean over all positions of -ln(p of the true next token).
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss


def build_model(cfg: dict, which: str):
    """Return (model, tokenizer) for Model "A" or Model "B"."""
    from src.utils import set_seed

    m = cfg["model"]
    set_seed(cfg["seed"])
    if which == "A":
        from src.embeddings import TokenEmbedding
        from src.tokenizer import BookTokenizer
        tokenizer = BookTokenizer.load(cfg["tokenizer"]["path"])
        embedding = TokenEmbedding(tokenizer.vocab_size, m["embedding_dimension"])
    elif which == "B":
        from src.gemma_embeddings import GemmaEmbedding, load_model_b_parts
        tokenizer, table = load_model_b_parts(cfg)
        embedding = GemmaEmbedding(table, m["embedding_dimension"], cfg["model_b"]["freeze_table"])
    elif which == "C":
        # Model C: the Model A tokenizer; Gemma rows (or the random control rows) -> P -> 256.
        from src.gemma_embeddings import GemmaEmbedding
        from src.tokenizer import BookTokenizer
        tokenizer = BookTokenizer.load(cfg["tokenizer"]["path"])
        table = torch.load(cfg["model_c"]["table_path"], weights_only=False)["table"]
        assert table.shape[0] == tokenizer.vocab_size, f"{cfg['model_c']['table_path']} is not a Model C table"
        embedding = GemmaEmbedding(table, m["embedding_dimension"], cfg["model_c"]["freeze_table"])
    else:
        raise ValueError(f"unknown model {which!r}: use A, B or C")
    untied = tokenizer.vocab_size if which == "C" and cfg["model_c"].get("untied_output") else None
    # A second seed for the blocks: the 6 layers start with the same numbers in Model A and Model B.
    # So a difference after training comes from the embedding only.
    set_seed(cfg["seed"] + 1)
    return BookTransformer(embedding, m, untied), tokenizer


if __name__ == "__main__":
    # Lesson 7 check: run with  python -m src.model
    from src.dataset import get_batch, load_token_ids
    from src.utils import load_config, set_seed, setup_console

    setup_console()
    cfg = load_config()
    m = cfg["model"]
    models = {}
    for which in ("A", "B", "C"):
        model, tokenizer = build_model(cfg, which)
        model.eval()                                     # dropout off: the same loss each run
        models[which] = model
        V = tokenizer.vocab_size
        train_ids, _ = load_token_ids(cfg, tokenizer)
        set_seed(cfg["seed"])
        x, y = get_batch(train_ids, cfg["training"]["batch_size"], m["context_length"])

        logits, loss = model(x, y)
        sigma = logits.std().item()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"=== Model {which} ===")
        print(f"Vocabulary:        {V:,}")
        print(f"Input IDs:         {list(x.shape)}")
        print(f"Logits:            {list(logits.shape)}")
        print(f"Trainable params:  {trainable:,}")
        print(f"Logit std (sigma): {sigma:.4f}")
        print(f"ln(V):             {math.log(V):.4f}")
        print(f"ln(V) + sigma^2/2: {math.log(V) + sigma ** 2 / 2:.4f}   (prediction for random logits)")
        print(f"Loss:              {loss.item():.4f}")
        assert list(logits.shape) == [x.shape[0], x.shape[1], V]
        assert abs(loss.item() - math.log(V)) < 0.2, "loss before training must be near ln(V)"

        # Weight tying: the output uses the embedding table. There is no separate output matrix.
        loss.backward()
        emb = model.embedding
        if which == "A":
            tied = emb.output_table() is emb.weight
            print(f"Output table is E (same tensor):  {tied}")
            assert tied
        else:
            print(f"Gemma table gradient:  {emb.table.grad}   P gradient shape: {list(emb.P.weight.grad.shape)}")
            assert emb.table.grad is None and emb.P.weight.grad is not None
        others = [n for n, p in model.named_parameters() if "embedding" not in n and p.shape[0] == V]
        if model.output is not None:
            # Untied Model C: exactly one other [V, 256] matrix, the output head.
            print(f"Untied output head:                {list(model.output.weight.shape)}")
            assert others == ["output.weight"] and list(model.output.weight.shape) == [V, m["embedding_dimension"]]
        else:
            print(f"No other [V, ...] matrix:          {not others}")
            assert not others
        print()

    for other in ("B", "C"):
        same = all(torch.equal(pa, pb) for pa, pb in zip(models["A"].blocks.parameters(), models[other].blocks.parameters()))
        print(f"Model A and Model {other} start with the same 6 blocks: {same}")
        assert same
