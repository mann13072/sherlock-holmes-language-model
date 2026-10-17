"""
Lesson 8: Training. The loop that changes the parameters to decrease the loss.

One training step:
    1. get a batch          input [16, 128], target [16, 128]
    2. forward pass         logits [16, 128, V] -> loss (one number)
    3. loss.backward()      a gradient for each parameter
    4. clip the gradient    the total length of all gradients is at most 1.0
    5. optimizer.step()     AdamW changes each parameter a small amount against its gradient

Model A and Model B read the same book text at each step. The two tokenizers differ,
so the script picks a character position in the book, then finds the token at that position.
Because the tokens differ, we compare the loss per character, not the loss per token.

Run from the project root:
    python train.py --memorize        Lesson 8 check: Model A and Model B memorize one batch
    python train.py --model A         train Model A
    python train.py --model B         train Model B
    python train.py --model A --resume artifacts/checkpoints/model_a/best.pt --add-book data/memoirs.txt
                                      continue Model A with a second book (new folder model_a_plus_memoirs)
"""

import argparse
import math
import os
import random
import re
import time

import torch
import torch.nn.functional as F

from src.dataset import load_book, split_train_val
from src.model import build_model
from src.utils import get_device, load_config, set_seed, setup_console

CHAR_MARGIN = 2000   # the last characters of the train text are never a window start (the window must fit)

# A sentence ends at . ! ? (also with a closing quote after it), but not after Mr. Mrs. Dr. St.
SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][”’]))(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)\s+")


def shuffle_sentences(text: str, seed: int) -> str:
    """Cut the text into sentences, put them in a random order, and join them with one space.

    Paragraph breaks are lost: they become sentence ends. The words in each sentence do not change.
    """
    sentences = [x for x in SENTENCE_END.split(text) if x]
    shuffled = sentences[:]
    random.Random(seed).shuffle(shuffled)
    assert sorted(shuffled) == sorted(sentences), "the shuffle must keep every sentence"
    return " ".join(shuffled)


def token_char_starts(tokenizer, text: str) -> list[int]:
    """The character position in `text` where each token ID of tokenizer.encode(text) starts."""
    if hasattr(tokenizer, "gemma"):
        # Model B: one Gemma token can become more than one book ID (a split piece). All pieces get its start.
        enc = tokenizer.gemma.encode(text, add_special_tokens=False)
        starts = []
        for g, (s, _) in zip(enc.ids, enc.offsets):
            n = 1 if g in tokenizer.to_book else len(tokenizer.split_piece(tokenizer.gemma.id_to_token(g)))
            starts.extend([s] * n)
        return starts
    return [s for s, _ in tokenizer.tok.encode(text).offsets]


class BookStream:
    """The token IDs of one text, with the character position of each token."""

    def __init__(self, tokenizer, text: str):
        self.text = text
        self.ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.starts = torch.tensor(token_char_starts(tokenizer, text), dtype=torch.long)
        assert len(self.ids) == len(self.starts), "each token ID must have a character position"

    def chars_of(self, first: int, last: int) -> int:
        """Number of characters in tokens first .. last-1."""
        end = self.starts[last].item() if last < len(self.starts) else len(self.text)
        return end - self.starts[first].item()


def batch_at_chars(stream: BookStream, char_positions: torch.Tensor, T: int):
    """One window for each character position. Return (input, target, characters in the targets)."""
    token_starts = torch.searchsorted(stream.starts, char_positions)
    token_starts = token_starts.clamp(max=len(stream.ids) - T - 1)
    x = torch.stack([stream.ids[i: i + T] for i in token_starts])
    y = torch.stack([stream.ids[i + 1: i + T + 1] for i in token_starts])
    chars = sum(stream.chars_of(i + 1, i + T + 1) for i in token_starts.tolist())
    return x, y, chars


def random_char_positions(generator: torch.Generator, n: int, text_length: int) -> torch.Tensor:
    """The same generator seed gives the same positions for Model A and Model B."""
    return torch.randint(text_length - CHAR_MARGIN, (n,), generator=generator)


def learning_rate(step: int, t: dict) -> float:
    """Warmup: a straight line from 0 to the peak. Then a cosine curve from the peak to the floor."""
    peak, floor = t["learning_rate"], t["min_learning_rate"]
    if step < t["warmup_steps"]:
        return peak * (step + 1) / t["warmup_steps"]
    progress = min(1.0, (step - t["warmup_steps"]) / max(1, t["max_steps"] - t["warmup_steps"]))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def make_optimizer(model, t: dict) -> torch.optim.AdamW:
    """AdamW. Weight decay pulls each matrix toward zero. The RMSNorm gains (1-D) get no decay."""
    params = [p for p in model.parameters() if p.requires_grad]   # Model B: the frozen Gemma table is not here
    groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": t["weight_decay"]},
              {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=t["learning_rate"], betas=tuple(t["betas"]))


@torch.no_grad()
def full_loss(model, stream: BookStream, T: int, batch_size: int) -> tuple[float, float]:
    """Loss on the whole text: every token except the first is a target one time.

    Return (loss per token, loss per character) = total loss / tokens, total loss / characters.
    """
    model.eval()
    device = next(model.parameters()).device
    n = len(stream.ids)
    window_starts = list(range(0, n - 1, T))           # windows side by side; the last window can be short
    total, count = 0.0, 0
    for k in range(0, len(window_starts), batch_size):
        for s in window_starts[k: k + batch_size]:
            x = stream.ids[s: min(s + T, n - 1)].unsqueeze(0).to(device)
            y = stream.ids[s + 1: min(s + T, n - 1) + 1].unsqueeze(0).to(device)
            logits, _ = model(x)
            total += F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), reduction="sum").item()
            count += y.numel()
    assert count == n - 1
    model.train()
    return total / count, total / stream.chars_of(1, n)


@torch.no_grad()
def sample_loss(model, stream: BookStream, positions: torch.Tensor, T: int, batch_size: int):
    """Loss on fixed windows of the train text (dropout off). Return (per token, per character)."""
    model.eval()
    total, count, chars = 0.0, 0, 0
    for k in range(0, len(positions), batch_size):
        x, y, c = batch_at_chars(stream, positions[k: k + batch_size], T)
        x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
        logits, _ = model(x)
        total += F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), reduction="sum").item()
        count += y.numel()
        chars += c
    model.train()
    return total / count, total / chars


@torch.no_grad()
def sample_text(model, tokenizer, prompt: str, n_tokens: int, T: int) -> str:
    """A short sample: draw each next token from the softmax probabilities (no top-k, no KV cache)."""
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=next(model.parameters()).device)
    for _ in range(n_tokens):
        logits, _ = model(ids[:, -T:])
        probs = torch.softmax(logits[0, -1], dim=-1)
        ids = torch.cat([ids, torch.multinomial(probs, 1).view(1, 1)], dim=1)
    model.train()
    return tokenizer.decode(ids[0].tolist())


def save_checkpoint(path: str, model, optimizer, cfg: dict, which: str, step: int, val_loss: float) -> None:
    from src.lora import has_lora, merged_state_dict
    if has_lora(model):
        # LoRA: save the merged weights (W + scale·B·A) with the keys of a plain model. The optimizer state belongs to
        # A and B only, so a later --resume starts a new optimizer.
        torch.save({"model": merged_state_dict(model), "optimizer": None, "config": cfg,
                    "which": which, "step": step, "val_loss": val_loss}, path)
        return
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg,
                "which": which, "step": step, "val_loss": val_loss}, path)


def train(cfg: dict, which: str, resume: str | None = None, extra_books: list[str] = (),
          shuffle: bool = False, extra_steps: int | None = None, run_name: str | None = None,
          restart_peak: float | None = None, epochs: float | None = None,
          schedule_steps: int | None = None, train_texts: list[str] = ()) -> None:
    """Train one model. With `resume`, continue from a checkpoint (weights, optimizer state, step).

    With `extra_books`, each book is split 90/10 by itself: the train parts are joined,
    and each validation part gets its own loss. The tokenizer does not change.

    With `shuffle`, the sentences of the joined train text get a random order. The validation text stays in order.
    With `extra_steps`, the run ends at (start step + extra_steps). After max_steps the learning rate stays at the floor.
    """
    t, m = cfg["training"], cfg["model"]
    T, B = m["context_length"], t["batch_size"]
    model, tokenizer = build_model(cfg, which)
    device = get_device()                                         # GPU on Colab, CPU on the PC
    model.to(device)
    print(f"Device: {device}")
    paths = [cfg["data"]["book_path"]] + list(extra_books)
    train_parts, val_streams = [], {}
    for path in paths:
        part_train, part_val = split_train_val(load_book(path), cfg["data"]["train_fraction"])
        train_parts.append(part_train)
        val_streams[os.path.splitext(os.path.basename(path))[0]] = BookStream(tokenizer, part_val)
    # --add-train-text: the whole file is train text, with no validation part. It goes after the first book:
    # at the end, the last CHAR_MARGIN characters would never start a window.
    train_parts[1:1] = [load_book(p) for p in train_texts]
    if cfg["data"].get("train_text_only"):
        # --train-text-only: train on the --add-train-text files alone. The books still give the validation parts.
        train_parts = [load_book(p) for p in train_texts]
    train_text = "\n\n".join(train_parts)
    if shuffle:
        ordered_chars = len(re.sub(r"\s", "", train_text))
        train_text = shuffle_sentences(train_text, cfg["seed"])
        assert len(re.sub(r"\s", "", train_text)) == ordered_chars, "the shuffle must keep every character"
        print(f"Shuffled {len(SENTENCE_END.split(train_text)):,} sentences. Start of the new train text:")
        print("  " + repr(train_text[:300]))
    train_stream = BookStream(tokenizer, train_text)
    if epochs:
        # One epoch = as many tokens as the train text has: train tokens / (batch_size x context_length).
        extra_steps = math.ceil(epochs * len(train_stream.ids) / (B * T * t["gradient_accumulation_steps"]))
        print(f"{epochs} epoch(s) = {extra_steps} steps")

    optimizer = make_optimizer(model, t)
    start_step = 0
    eval_every = t["eval_interval"]
    if epochs or schedule_steps:
        # --epochs or --schedule-steps: evaluate at the end of each epoch, not every eval_interval steps.
        eval_every = math.ceil(len(train_stream.ids) / (B * T * t["gradient_accumulation_steps"]))
    end_step = t["max_steps"]
    if resume:
        ckpt = torch.load(resume, weights_only=False, map_location=device)
        assert ckpt["which"] == which, f"{resume} is a checkpoint of Model {ckpt['which']}"
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:                     # a LoRA checkpoint has no optimizer state
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Continue from {resume} (step {start_step}, validation loss {ckpt['val_loss']:.3f})")
    if t.get("lora_rank"):
        # --lora-rank: freeze all weights; only the LoRA matrices A and B of the 7 matrices of each block learn.
        from src.lora import add_lora
        n_lora = add_lora(model, t["lora_rank"], t["lora_alpha"])
        model.to(device)
        optimizer = make_optimizer(model, t)                      # a new optimizer for A and B only
        print(f"LoRA rank {t['lora_rank']}, alpha {t['lora_alpha']}: {n_lora:,} trainable numbers; all weights frozen")
    end_step = start_step + extra_steps if extra_steps else t["max_steps"]
    if schedule_steps:
        # --schedule-steps N: the run follows the config warmup and cosine over steps 0..N, and ends at step N.
        # A resumed run then continues the same curve, as if it never stopped.
        end_step = schedule_steps
    # The checkpoint config records the data, so a later script can make the same text again.
    cfg["data"]["extra_book_paths"] = list(extra_books)
    cfg["data"]["train_text_paths"] = list(train_texts)
    cfg["training"]["resumed_from"] = resume
    cfg["data"]["shuffle_sentences"] = shuffle
    # --restart-peak: a new schedule for the extra steps (100 warmup steps, then cosine to the floor).
    schedule, offset = t, 0
    if restart_peak:
        assert extra_steps, "a new schedule needs --extra-steps"
        schedule = dict(t, learning_rate=restart_peak, warmup_steps=100, max_steps=extra_steps)
        offset = start_step
    elif schedule_steps:
        schedule = dict(t, max_steps=schedule_steps)
    elif extra_steps and not resume:
        # A fresh run with --epochs or --extra-steps: the config warmup and cosine cover the whole run.
        schedule = dict(t, max_steps=extra_steps)
    cfg["training"]["restart_peak"] = restart_peak
    set_seed(cfg["seed"])                                         # dropout
    batch_gen = torch.Generator().manual_seed(cfg["seed"])        # batch positions: the same for A and B
    # The train-loss sample windows do not change with --seed, so the train/validation gap compares across seeds.
    eval_gen = torch.Generator().manual_seed(cfg.get("eval_seed", cfg["seed"]) + 100)
    eval_positions = random_char_positions(eval_gen, t["eval_batches"] * B, len(train_text))

    run = which + "".join(f"_plus_{os.path.splitext(os.path.basename(p))[0]}" for p in extra_books)
    run += "_shuffled" if shuffle else ""
    run = run_name or run                                         # --run-name: a folder and a log of its own
    out_dir = os.path.join(t["checkpoint_dir"], f"model_{run.lower()}")
    assert not (resume and os.path.abspath(out_dir) == os.path.abspath(os.path.dirname(resume))), \
        "a continued run must not write into the folder of its start checkpoint"
    os.makedirs(out_dir, exist_ok=True)
    # No CSV log (the user decided on 2026-10-16). The printed lines are the log: colab_run.sh saves them
    # in artifacts/logs/train_output_<run>.txt. On the PC, redirect the output to that file.

    print(f"Model {which}: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    print(f"Train tokens {len(train_stream.ids):,}   validation tokens "
          f"{', '.join(f'{k} {len(v.ids):,}' for k, v in val_streams.items())}   "
          f"one epoch = {len(train_stream.ids) / (B * T):.0f} steps")
    best_val, evals_without_gain = float("inf"), 0
    model.train()
    run_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(start_step, end_step + 1):
        # Evaluate at the start step (before any change), then every eval_every steps (or each epoch).
        if (step - start_step) % eval_every == 0 or step == end_step:
            tr_tok, tr_chr = sample_loss(model, train_stream, eval_positions, T, B)
            per_book = {k: full_loss(model, v, T, B) for k, v in val_streams.items()}
            # All validation books together: total loss / total tokens, total loss / total characters.
            counts = {k: len(v.ids) - 1 for k, v in val_streams.items()}
            chars = {k: v.chars_of(1, len(v.ids)) for k, v in val_streams.items()}
            total = sum(per_book[k][0] * counts[k] for k in val_streams)
            va_tok, va_chr = total / sum(counts.values()), total / sum(chars.values())
            books = "".join(f" | {k} {tok:.3f} ({chr_:.3f}/char)" for k, (tok, chr_) in per_book.items()) \
                if len(val_streams) > 1 else ""
            print(f"step {step:5d} | train {tr_tok:.3f} ({tr_chr:.3f}/char) | val {va_tok:.3f} "
                  f"({va_chr:.3f}/char, perplexity {math.exp(va_tok):7.1f}){books} | lr {learning_rate(step - offset, schedule):.2e}",
                  flush=True)
            if va_tok < best_val:
                best_val, evals_without_gain = va_tok, 0
                save_checkpoint(os.path.join(out_dir, "best.pt"), model, optimizer, cfg, which, step, va_tok)
            else:
                evals_without_gain += 1
            if step > start_step and (step % t["checkpoint_interval"] == 0 or step == end_step):   # also the last step
                save_checkpoint(os.path.join(out_dir, f"step_{step}.pt"), model, optimizer, cfg, which, step, va_tok)
                print("  sample:", repr(sample_text(model, tokenizer, t["sample_prompt"], t["sample_tokens"], T)))
            if evals_without_gain >= t["early_stop_patience"]:
                print(f"Stop: the validation loss did not improve in {evals_without_gain} evaluations "
                      f"(best {best_val:.3f}).")
                break
        if step == end_step:
            break

        # A file named STOP in the project folder ends the run: save the last step, delete STOP, and stop.
        if step % 10 == 0 and os.path.exists("STOP"):
            save_checkpoint(os.path.join(out_dir, f"step_{step}.pt"), model, optimizer, cfg, which, step, float("nan"))
            os.remove("STOP")
            print(f"STOP file found at step {step}: saved step_{step}.pt and stopped.", flush=True)
            break

        # --dropout-end: the dropout falls in equal steps, once at the start of each epoch, from --dropout to this value.
        if t.get("dropout_end") is not None and (step - start_step) % eval_every == 0:
            n_epochs = max(1, math.ceil((end_step - start_step) / eval_every))
            epoch = (step - start_step) // eval_every
            p = cfg["model"]["dropout"] + (t["dropout_end"] - cfg["model"]["dropout"]) * epoch / max(1, n_epochs - 1)
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = p
            print(f"  dropout {p:.4f} in epoch {epoch + 1} of {n_epochs}", flush=True)

        # ---- one training step ----
        lr = learning_rate(step - offset, schedule)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        for _ in range(t["gradient_accumulation_steps"]):
            x, y, _ = batch_at_chars(train_stream, random_char_positions(batch_gen, B, len(train_text)), T)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / t["gradient_accumulation_steps"]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"])
        optimizer.step()
    print(f"Best validation loss {best_val:.3f}. Checkpoints in {out_dir}.")
    peak = f"{torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB" if device.type == "cuda" else "no GPU"
    print(f"Training time {time.time() - run_start:,.0f} s (with evaluations). Peak GPU memory {peak}.")


def memorize_one_batch(cfg: dict, max_steps: int = 300, target: float = 0.1) -> None:
    """Lesson 8 check: with one fixed batch, the loss must go near 0.

    If the model cannot memorize 2,048 predictions, the training loop has a bug.
    Dropout is off and the learning rate is constant, so only the loop is tested.
    """
    t, m = cfg["training"], cfg["model"]
    T, B = m["context_length"], t["batch_size"]
    text = load_book(cfg["data"]["book_path"])
    train_text, _ = split_train_val(text, cfg["data"]["train_fraction"])
    positions = random_char_positions(torch.Generator().manual_seed(cfg["seed"]), B, len(train_text))
    first_text = {}
    for which in ("A", "B"):
        model, tokenizer = build_model(cfg, which)
        stream = BookStream(tokenizer, train_text)
        x, y, chars = batch_at_chars(stream, positions, T)
        token0 = torch.searchsorted(stream.starts, positions[:1]).item()
        first_text[which] = (stream.starts[token0].item(), tokenizer.decode(x[0, :12].tolist()))
        model.eval()                                   # dropout off
        optimizer = make_optimizer(model, t)
        start = time.time()
        for step in range(max_steps + 1):
            _, loss = model(x, y)
            if step % 25 == 0 or loss.item() < target:
                print(f"Model {which}  step {step:3d}  loss {loss.item():.4f}", flush=True)
            if loss.item() < target:
                break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"])
            optimizer.step()
        print(f"Model {which}: loss {loss.item():.4f} after {step} steps ({time.time() - start:.0f} s). "
              f"Window 0 starts at character {first_text[which][0]:,}: {first_text[which][1]!r}\n")
        assert loss.item() < target, f"Model {which} did not memorize one batch"
    # The two models read the same book text: window 0 starts at almost the same character.
    assert abs(first_text["A"][0] - first_text["B"][0]) < 20


if __name__ == "__main__":
    setup_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["A", "B", "C"])
    parser.add_argument("--memorize", action="store_true", help="Lesson 8 check: memorize one batch")
    parser.add_argument("--resume", help="continue from this checkpoint, for example artifacts/checkpoints/model_a/best.pt")
    parser.add_argument("--add-book", action="append", default=[], help="one more book for training (can repeat)")
    parser.add_argument("--shuffle-sentences", action="store_true", help="random order of the train sentences")
    parser.add_argument("--extra-steps", type=int, help="train this many steps after the resume step")
    parser.add_argument("--run-name", help="the name of the checkpoint folder and the log (default: from the books)")
    parser.add_argument("--restart-peak", type=float, help="new warmup + cosine for the extra steps, with this peak")
    parser.add_argument("--epochs", type=float, help="train this many epochs after the resume step (instead of --extra-steps)")
    parser.add_argument("--schedule-steps", type=int,
                        help="the config warmup + cosine over steps 0..N; the run ends at step N (use with --resume to continue a run)")
    parser.add_argument("--gemma-table", help="Model B or C: use this table file instead of model_b/model_c.table_path")
    parser.add_argument("--train-gemma-table", action="store_true",
                        help="Model B or C: the optimizer also changes the table rows (not only P)")
    parser.add_argument("--layers", type=int, help="number of Transformer layers (default: the config preset)")
    parser.add_argument("--context-length", type=int, help="tokens in one window (default: the config preset, 128)")
    parser.add_argument("--batch-size", type=int, help="windows in one step (default: config, 16)")
    parser.add_argument("--add-train-text", action="append", default=[],
                        help="a file that is only train text, with no validation part (can repeat)")
    parser.add_argument("--seed", type=int, help="seed for the start weights, dropout and batches (default: config)")
    parser.add_argument("--dropout", type=float, help="dropout probability (default: the config preset, 0.1)")
    parser.add_argument("--lr", type=float, help="peak learning rate of a fresh run (default: config, 3e-4)")
    parser.add_argument("--lora-rank", type=int, help="with --resume: LoRA of this rank; all weights stay frozen")
    parser.add_argument("--lora-alpha", type=float, help="LoRA scale = alpha / rank (default: alpha = 2 x rank)")
    parser.add_argument("--train-text-only", action="store_true",
                        help="train on the --add-train-text files only; the books give only the validation parts")
    parser.add_argument("--dropout-end", type=float,
                        help="with --epochs: the dropout falls once per epoch, in equal steps, from --dropout to this value")
    args = parser.parse_args()
    config = load_config()
    if args.layers:
        config["model"]["num_transformer_layers"] = args.layers   # the checkpoint config records it
    if args.context_length:
        # RoPE and the causal mask are made for each sequence length, so no weight changes shape.
        config["model"]["context_length"] = args.context_length   # the checkpoint config records it
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size        # the checkpoint config records it
    config["eval_seed"] = config["seed"]                        # the train-loss windows always use the config seed
    if args.seed is not None:
        config["seed"] = args.seed                              # the checkpoint config records it
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout               # the checkpoint config records it
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr           # the peak; the floor stays min_learning_rate
    if args.lora_rank:
        assert args.resume, "--lora-rank needs --resume (LoRA adapts a trained model)"
        config["training"]["lora_rank"] = args.lora_rank        # the checkpoint config records it
        config["training"]["lora_alpha"] = args.lora_alpha or 2 * args.lora_rank
    if args.train_text_only:
        assert args.add_train_text, "--train-text-only needs --add-train-text"
        config["data"]["train_text_only"] = True                # the checkpoint config records it
    if args.dropout_end is not None:
        assert args.epochs, "--dropout-end needs --epochs (the dropout changes once per epoch)"
        config["training"]["dropout_end"] = args.dropout_end    # the checkpoint config records it
    table_key = "model_c" if args.model == "C" else "model_b"
    if args.gemma_table:
        config[table_key]["table_path"] = args.gemma_table   # the checkpoint config records it
    if args.train_gemma_table:
        config[table_key]["freeze_table"] = False             # the checkpoint config records it
    if args.memorize:
        memorize_one_batch(config)
    else:
        train(config, args.model, args.resume, args.add_book, args.shuffle_sentences, args.extra_steps,
              args.run_name, args.restart_peak, args.epochs, args.schedule_steps, args.add_train_text)
