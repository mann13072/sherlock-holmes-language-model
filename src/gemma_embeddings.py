"""
Model B: the pretrained token table of EmbeddingGemma 2, cut down to the book.

    text -> Gemma tokenizer -> Gemma IDs (0 .. 262,143)
         -> book IDs (0 .. N-1)            only the Gemma tokens of the train text + 256 bytes
         -> row of the Gemma table [512]   pretrained, frozen
         -> P (512 -> 256)                 learned
         -> vector [256]                   the same size as in Model A

Model A (src/embeddings.py) starts from random numbers. Model B starts from
the table that Google trained on a very large amount of text. The 6 layers
are the same in both models.

The full Gemma table has 262,144 rows. The model needs one logit for each row
(weight tying), so we keep only the N rows that the book uses.

A validation token that is not in the train text becomes known pieces
("▁lunatic" -> "▁l" + "un" + "atic"). A character with no known piece becomes
its UTF-8 bytes. So the validation text stays unseen, as in Model A.

Model C: the Model A tokenizer (4,000 tokens) with Gemma rows. Each Model A token goes through the
Gemma tokenizer; its row is the Gemma row (1 piece) or the mean of the Gemma rows (2+ pieces).
P (512 -> 256) learns, as in Model B. The control table has random rows with the same std.

Prepare the table one time:  python -m src.gemma_embeddings --prepare
Model C tables:              python -m src.gemma_embeddings --prepare-c   (add --random for the control)
Run the checks:              python -m src.gemma_embeddings
"""

import math

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

GEMMA_SPACE = "▁"   # the Gemma tokenizer writes a space as this symbol


def load_gemma_tokenizer(repo: str) -> Tokenizer:
    return Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))


def byte_token_ids(gemma: Tokenizer) -> list[int]:
    """The Gemma IDs of the 256 byte tokens <0x00> .. <0xFF>, in byte order."""
    return [gemma.token_to_id(f"<0x{b:02X}>") for b in range(256)]


class GemmaBookTokenizer:
    """The Gemma tokenizer with a small vocabulary: book ID i = gemma_ids[i].

    It has the same operations as BookTokenizer, so load_token_ids and
    get_batch (src/dataset.py) work without a change.
    """

    def __init__(self, gemma: Tokenizer, gemma_ids: list[int]):
        self.gemma = gemma
        self.gemma_ids = gemma_ids
        self.to_book = {g: b for b, g in enumerate(gemma_ids)}
        self.byte_book_ids = [self.to_book[g] for g in byte_token_ids(gemma)]
        # Piece string -> book ID, for the split of an unknown token. Byte tokens are not pieces.
        byte_set = set(self.byte_book_ids)
        self.piece_to_book = {gemma.id_to_token(g): b for b, g in enumerate(gemma_ids) if b not in byte_set}
        self.longest_piece = max(len(p) for p in self.piece_to_book)
        self.split_count = 0   # Gemma tokens that the last encode split into known pieces
        self.byte_count = 0    # byte tokens that the last encode used (a character with no known piece)

    @property
    def vocab_size(self) -> int:
        return len(self.gemma_ids)

    def split_piece(self, piece: str) -> list[int]:
        """Split an unknown piece into known pieces: the longest match from the left.

        Example: "▁lunatic" is not in the train text. It becomes "▁l" + "un" + "atic".
        A character with no known piece becomes its UTF-8 bytes (the last option).
        """
        out, i = [], 0
        while i < len(piece):
            for j in range(min(len(piece), i + self.longest_piece), i, -1):
                if piece[i:j] in self.piece_to_book:
                    out.append(self.piece_to_book[piece[i:j]])
                    i = j
                    break
            else:
                char = piece[i].replace(GEMMA_SPACE, " ")
                out.extend(self.byte_book_ids[b] for b in char.encode("utf-8"))
                self.byte_count += len(char.encode("utf-8"))
                i += 1
        return out

    def encode(self, text: str) -> list[int]:
        """Text -> book IDs. A Gemma token outside the book vocabulary becomes known pieces."""
        # add_special_tokens=False: no <bos> at the start of each text.
        out, self.split_count, self.byte_count = [], 0, 0
        for g in self.gemma.encode(text, add_special_tokens=False).ids:
            if g in self.to_book:
                out.append(self.to_book[g])
            else:
                out.extend(self.split_piece(self.gemma.id_to_token(g)))
                self.split_count += 1
        return out

    def decode(self, ids: list[int]) -> str:
        """Book IDs -> text. The Gemma decoder joins the byte tokens again."""
        return self.gemma.decode([self.gemma_ids[i] for i in ids])

    def tokens(self, text: str) -> list[str]:
        """Text -> readable tokens. Each ID is decoded alone."""
        return [self.decode([i]) for i in self.encode(text)]


def prepare_table(cfg: dict, extra_books: list[str] = (), out_path: str | None = None) -> None:
    """Make the book vocabulary and copy its rows of the Gemma table to a small file.

    With `extra_books`, the vocabulary comes from the train parts of all books (each split 90/10 by itself),
    as in train.py. The file also stores the Gemma tokenizer, so Colab needs no download.
    """
    from safetensors import safe_open

    from src.dataset import load_book, split_train_val

    mb = cfg["model_b"]
    out_path = out_path or mb["table_path"]
    gemma = load_gemma_tokenizer(mb["gemma_repo"])
    paths = [cfg["data"]["book_path"]] + list(extra_books)
    train_text = "\n\n".join(split_train_val(load_book(p), cfg["data"]["train_fraction"])[0] for p in paths)

    # Only the train text selects the vocabulary, as for the Model A tokenizer (Lesson 3).
    train_ids = set(gemma.encode(train_text, add_special_tokens=False).ids)
    gemma_ids = sorted(train_ids | set(byte_token_ids(gemma)))   # sorted: the same order each time

    weights = hf_hub_download(mb["gemma_repo"], "model.safetensors")
    with safe_open(weights, framework="pt") as f:
        full = f.get_tensor(mb["gemma_tensor"])                      # [262144, 512], bfloat16
    table = full[gemma_ids].float()                                  # [N, 512], float32

    torch.save({"gemma_ids": torch.tensor(gemma_ids), "table": table,
                "repo": mb["gemma_repo"], "tensor": mb["gemma_tensor"],
                "books": paths, "tokenizer_json": gemma.to_str()}, out_path)
    print(f"Gemma table:         {list(full.shape)}  ({full.dtype})")
    print(f"Train tokens:        {len(train_ids):,} different Gemma tokens")
    added = len(gemma_ids) - len(train_ids)
    print(f"Byte tokens added:   {added} of 256  ({256 - added} occur in the train text)")
    print(f"Book table:          {list(table.shape)}")
    print(f"Saved:               {out_path}  ({table.numel() * 4 / 1e6:.1f} MB)")


def model_a_token_pieces(vocab_path: str, gemma: Tokenizer) -> list[list[int]]:
    """The Gemma IDs of each Model A token, in Model A ID order.

    A Model A token is byte-level BPE ("ĠHolmes" = b" Holmes"). Valid UTF-8 text goes through the
    Gemma tokenizer. A part of a character (for example 2 of the 3 bytes of “) gets the Gemma byte tokens.
    """
    import json

    # The inverse of the GPT-2 byte-level map: one visible character -> one byte.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    to_byte = {chr(c): b for b, c in zip(bs, cs)}

    saved = json.load(open(vocab_path, encoding="utf-8"))
    assert saved["added_tokens"] == []
    vocab = saved["model"]["vocab"]
    gemma_bytes = byte_token_ids(gemma)
    pieces = []
    for piece, _ in sorted(vocab.items(), key=lambda kv: kv[1]):
        raw = bytes(to_byte[ch] for ch in piece)
        try:
            text = raw.decode("utf-8")
            ids = gemma.encode(text, add_special_tokens=False).ids
            assert gemma.decode(ids) == text
        except UnicodeDecodeError:
            ids = [gemma_bytes[b] for b in raw]
        pieces.append(ids)
    return pieces


def prepare_model_c_table(cfg: dict, random_rows: bool = False, out_path: str | None = None) -> None:
    """Make the Model C table [4000, 512]: Gemma rows (mean over pieces), or random rows with the same std."""
    from safetensors import safe_open

    mb = cfg["model_b"]
    gemma = load_gemma_tokenizer(mb["gemma_repo"])
    pieces = model_a_token_pieces(cfg["tokenizer"]["path"], gemma)
    weights = hf_hub_download(mb["gemma_repo"], "model.safetensors")
    with safe_open(weights, framework="pt") as f:
        full = f.get_tensor(mb["gemma_tensor"])                      # [262144, 512], bfloat16
    table = torch.stack([full[ids].float().mean(dim=0) for ids in pieces])   # [4000, 512], float32

    # Check: 1 piece = the Gemma row itself; 2+ pieces = the mean of the Gemma rows.
    for ids, row in zip(pieces, table):
        assert torch.equal(row, full[ids].float().mean(dim=0))
        if len(ids) == 1:
            assert torch.equal(row, full[ids[0]].float())

    source = "gemma"
    if random_rows:
        # Control: the same shape and std, no Gemma content. Then only the Gemma content differs.
        gen = torch.Generator().manual_seed(cfg["seed"])
        table = torch.randn(table.shape, generator=gen) * table.std()
        source = "random"
    out_path = out_path or (cfg["model_c"]["table_path"] if not random_rows
                            else cfg["model_c"]["table_path"].replace(".pt", "_random.pt"))
    torch.save({"table": table, "pieces": pieces, "source": source,
                "repo": mb["gemma_repo"], "tensor": mb["gemma_tensor"]}, out_path)
    counts = [len(ids) for ids in pieces]
    print(f"Model A tokens:      {len(pieces):,}   1 piece {counts.count(1):,}   2+ pieces {len(pieces) - counts.count(1):,}")
    print(f"Table ({source}):    {list(table.shape)}   std {table.std():.5f}   mean row length {table.norm(dim=1).mean():.4f}")
    print(f"Saved:               {out_path}  ({table.numel() * 4 / 1e6:.1f} MB)")


def load_model_b_parts(cfg: dict) -> tuple[GemmaBookTokenizer, torch.Tensor]:
    """Return the Model B tokenizer and the book table [N, 512]."""
    saved = torch.load(cfg["model_b"]["table_path"])
    # A newer table file stores the Gemma tokenizer. An older one downloads it from Hugging Face.
    gemma = Tokenizer.from_str(saved["tokenizer_json"]) if "tokenizer_json" in saved \
        else load_gemma_tokenizer(saved["repo"])
    tokenizer = GemmaBookTokenizer(gemma, saved["gemma_ids"].tolist())
    return tokenizer, saved["table"]


class GemmaEmbedding(nn.Module):
    """Book ID -> row of the Gemma table [512] -> P -> vector [embedding_dimension].

    The forward pass has the same shapes as TokenEmbedding:
    [batch, seq] -> [batch, seq, embedding_dimension].
    """

    def __init__(self, table: torch.Tensor, embedding_dimension: int, freeze_table: bool = True):
        super().__init__()
        # Frozen: the table is a buffer. The optimizer does not change it.
        # The model must use the Gemma vectors as they are, and only P learns.
        if freeze_table:
            self.register_buffer("table", table)
        else:
            self.table = nn.Parameter(table)

        # P changes 512 Gemma numbers into 256 numbers. It is the only new part.
        self.P = nn.Linear(table.shape[1], embedding_dimension, bias=False)

        # Random start for P. The std makes the output std about 0.02, the same
        # as the start of E in Model A. One output number = sum of 512 products,
        # so its variance = 512 x std(P)^2 x mean(row^2).
        std = 0.02 / math.sqrt(table.shape[1] * table.pow(2).mean().item())
        nn.init.normal_(self.P.weight, mean=0.0, std=std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # [batch, seq] -> [batch, seq, 512] -> [batch, seq, 256]
        return self.P(self.table[token_ids])

    def output_table(self) -> torch.Tensor:
        """The table after P, [N, embedding_dimension]. The output layer uses it (weight tying)."""
        return self.P(self.table)


if __name__ == "__main__":
    import argparse

    from src.utils import load_config, set_seed, setup_console

    setup_console()
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="make artifacts/gemma_book_table.pt")
    parser.add_argument("--add-book", action="append", default=[], help="with --prepare: one more book (can repeat)")
    parser.add_argument("--out", help="with --prepare: the table file (default: model_b.table_path)")
    parser.add_argument("--prepare-c", action="store_true", help="make the Model C table (model_c.table_path)")
    parser.add_argument("--random", action="store_true", help="with --prepare-c: the random control table")
    args = parser.parse_args()
    if args.prepare:
        prepare_table(cfg, args.add_book, args.out)
        raise SystemExit
    if args.prepare_c:
        prepare_model_c_table(cfg, args.random, args.out)
        raise SystemExit

    from src.attention import CausalSelfAttention
    from src.dataset import get_batch, load_book, load_token_ids, split_train_val
    from src.tokenizer import BookTokenizer

    set_seed(cfg["seed"])
    m = cfg["model"]
    D = m["embedding_dimension"]
    tokenizer, table = load_model_b_parts(cfg)
    N = tokenizer.vocab_size

    # --- Lesson 3 for Model B: the tokenizer
    print("=== Lesson 3: tokenizer ===")
    text = load_book(cfg["data"]["book_path"])
    train_text, val_text = split_train_val(text, cfg["data"]["train_fraction"])
    book_a = BookTokenizer.load(cfg["tokenizer"]["path"])
    print(f"Vocabulary:  Model A {book_a.vocab_size:,}   Model B {N:,}  (of 262,144 Gemma tokens)")
    sentence = "To Sherlock Holmes she is always _the_ woman."
    print(f"Model A tokens: {book_a.tokens(sentence)}")
    print(f"Model B tokens: {tokenizer.tokens(sentence)}")
    for name, t in [("train", train_text), ("validation", val_text)]:
        ids_a, ids_b = book_a.encode(t), tokenizer.encode(t)
        ok = tokenizer.decode(ids_b) == t
        print(f"{name:>10}: decode(encode(text)) == text: {ok}   unknown Gemma tokens split: {tokenizer.split_count:,}"
              f"   byte tokens: {tokenizer.byte_count:,}"
              f"   characters per token: A {len(t) / len(ids_a):.2f}, B {len(t) / len(ids_b):.2f}")
        assert ok
    print()

    # --- Lesson 4 for Model B: training examples (the same get_batch)
    print("=== Lesson 4: training examples ===")
    train_ids, val_ids = load_token_ids(cfg, tokenizer)
    x, y = get_batch(train_ids, cfg["training"]["batch_size"], m["context_length"])
    shifted = torch.equal(y[:, :-1], x[:, 1:])
    print(f"Train IDs: {len(train_ids):,}   validation IDs: {len(val_ids):,}   input {list(x.shape)}")
    print(f"target[:, :-1] == input[:, 1:]  ->  {shifted}")
    assert shifted

    # A validation target that never occurs in the train IDs: training pushes its probability to almost 0.
    # The count must be small for both models, or the validation loss is not fair.
    for name, tok in [("Model A", book_a), ("Model B", tokenizer)]:
        tr, va = load_token_ids(cfg, tok)
        never = (torch.bincount(tr, minlength=tok.vocab_size)[va] == 0).sum().item()
        print(f"{name}: validation tokens that never occur in train: {never:,} of {len(va):,}")
    print()

    # --- Lesson 5a for Model B: the embedding
    print("=== Lesson 5a: embedding ===")
    set_seed(cfg["seed"])   # the same start of P as in the story GIF
    emb = GemmaEmbedding(table, D, cfg["model_b"]["freeze_table"])
    trainable = sum(p.numel() for p in emb.parameters() if p.requires_grad)
    print(f"Gemma table: {list(emb.table.shape)}   frozen: {not emb.table.requires_grad}")
    print(f"P:           {list(emb.P.weight.shape)}   trainable parameters: {trainable:,}"
          f"   (Model A E: {book_a.vocab_size * D:,})")
    print(f"Row std of the Gemma table: {table.std().item():.4f}   mean row length: {table.norm(dim=1).mean().item():.3f}")
    out_table = emb.output_table()
    print(f"Std of P x row (all {N:,} rows): {out_table.std().item():.4f}   (Model A start: 0.02)")
    assert abs(out_table.std().item() - 0.02) < 0.002

    ids = torch.tensor([tokenizer.encode(sentence)])
    vec = emb(ids)
    k = tokenizer.tokens(sentence).index(" Holmes")
    bid = ids[0, k].item()
    by_hand = emb.P.weight @ table[bid]
    print(f"' Holmes' = book ID {bid} = Gemma ID {tokenizer.gemma_ids[bid]}")
    print(f"  table row [512] -> vector {list(vec[0, k].shape)}: {[round(v, 4) for v in vec[0, k, :4].tolist()]} ...")
    print(f"  lookup == P x (row of the table): {torch.allclose(vec[0, k], by_hand, atol=1e-6)}")
    print(f"  lookup == row of output_table:    {torch.allclose(vec[0, k], out_table[bid], atol=1e-6)}")
    assert torch.allclose(vec[0, k], by_hand, atol=1e-6)

    vec.sum().backward()
    print(f"  after backward(): table gradient = {emb.table.grad}   P gradient shape = {list(emb.P.weight.grad.shape)}")
    assert emb.table.grad is None and emb.P.weight.grad is not None
    print()

    # --- Lessons 5b and 6 for Model B: RoPE and attention see only 256 numbers per token
    print("=== Lessons 5b + 6: attention with Model B vectors ===")
    attn = CausalSelfAttention(D, m["num_attention_heads"], m["head_dimension"], m["dropout"]).eval()

    def rmsnorm(h):  # the planned formula; src/rmsnorm.py comes in Lesson 7
        return h / torch.sqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    T = ids.shape[1]
    with torch.no_grad():
        out = attn(rmsnorm(emb(ids)))
        ids2 = ids.clone()
        ids2[0, T - 1] = tokenizer.encode("!")[0]
        out2 = attn(rmsnorm(emb(ids2)))
    change = (out2 - out).abs().amax(dim=-1)[0]
    print(f"Attention output: {list(out.shape)}")
    print(f"Change the last token '.' -> '!'. Positions 0..{T - 2} unchanged: {bool((change[:-1] == 0).all())}"
          f"   last position changed: {bool(change[-1] > 0)}")
    assert (change[:-1] == 0).all() and change[-1] > 0
