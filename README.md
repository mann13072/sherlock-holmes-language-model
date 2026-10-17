# Sherlock Holmes Language Model

A very small Transformer language model, built from zero to learn how language models work.
It trains on the 9 Sherlock Holmes books, Wikipedia articles about the stories, and related public text.

The goal of the project is understanding. Model quality is secondary.

Video: [`portfolio/make_video.mp4`](portfolio/make_video.mp4). All numbers in the video come from the checkpoint in this repository.

## The model

| Part | Value | File |
|---|---|---|
| Type | Decoder-only Transformer (the style of Llama-type models, very small) | `src/model.py` |
| Tokenizer | Byte-level BPE, 4,000 tokens, trained on *The Adventures of Sherlock Holmes* | `artifacts/tokenizer.json`, `src/tokenizer.py` |
| Input embedding | A frozen table [4000, 512] from the EmbeddingGemma 2 input rows, then a learned projection P (512 → 256) | `src/gemma_embeddings.py` |
| Blocks | 6 pre-norm blocks: RMSNorm → attention → add; RMSNorm → SwiGLU MLP (width 1024) → add | `src/transformer_block.py` |
| Attention | 8 heads × 32 dimensions, causal mask, calculated step by step (no fused kernel) | `src/attention.py` |
| Positions | RoPE on each head; θᵢ = 10000^(−2i/32) | `src/rope.py` |
| Output | Final RMSNorm, then a separate output head (256 → 4000) | `src/model.py` |
| Context | 256 tokens | `config.yaml` |
| Trainable parameters | 7,449,856 | |

## Results

The validation set is the last 10% of each of the 13 source texts (9 books, 4 article sets). Lower is better.

| Model | Validation loss |
|---|---|
| First model, own embedding table, 10 epochs, dropout 0.1 | 3.618 |
| The same, dropout 0.2, 20 epochs | 3.520 |
| Frozen Gemma table + separate output head, context 128 | 3.513 |
| The same, context 256 | 3.499 |
| **The same, continued on new related text (this checkpoint)** | **3.410** (1.452 bits per character) |

Main findings:

- The largest gain came from new, different text. More epochs on the same text made the model worse.
- The model is short of data. 7.4 M parameters need about 150 M training tokens; this project has about 2 M.
- Context 256 helps only on long text. A model trained at 128 tokens cannot use positions 128–255.
- No method taught the model facts for questions that it did not see in training.

## Use the model

```bash
pip install -r requirements.txt
```

Run this from the repository root:

```python
import torch
from src.model import build_model

ck = torch.load("artifacts/checkpoints/model_c256_newdata_mix/best.pt", weights_only=False, map_location="cpu")
model, tok = build_model(ck["config"], ck["which"])      # which = "C"
model.load_state_dict(ck["model"])
model.eval()

with torch.no_grad():
    logits = model(torch.tensor([tok.encode("221B Baker")]))[0]
probs = torch.softmax(logits[0, -1], -1)
values, ids = probs.topk(3)
print([(tok.decode([i]), round(v, 3)) for v, i in zip(values.tolist(), ids.tolist())])
# [(' Street', 0.993), (' Holmes', 0.001), (' Yard', 0.001)]
```

NOTE: The checkpoint holds the weights, the config, the step and the validation loss. The optimizer state is not included.

## Train the model

The training text is not in this repository. To train again, put the text files in `data/`, with the names in `config.yaml`.

```bash
python train.py --model C --gemma-table artifacts/gemma_model_c_table.pt --context-length 256 --batch-size 8 \
    --epochs 20 --dropout 0.2 --lr 6e-4 --seed 42 --add-book data/memoirs.txt ...
```

The training used AdamW (betas 0.9/0.95, weight decay 0.1), a warmup of 200 steps, a cosine schedule to 3e-5, and a gradient clip of 1.0. It ran on a Colab T4 GPU.

## Training data (not included)

- The 9 Sherlock Holmes books by Arthur Conan Doyle, from Project Gutenberg (#1661, #834, #108, #2852, #2350, #2097, #244, #3289, #69700).
- 100 Wikipedia articles about the stories, actors, adaptations and fandom (CC BY-SA 4.0).
- Doyle: *Memories and Adventures*, *Through the Magic Door*.
- Essays and books about detective fiction, 48 Wikipedia articles about Victorian London and the police, and two histories of the English police.

## Credits

- The input table is made from the input-token embeddings of [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) by Google (Apache 2.0).
- Book texts: [Project Gutenberg](https://www.gutenberg.org).
