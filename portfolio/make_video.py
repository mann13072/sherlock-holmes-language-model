"""
Portfolio video of the Sherlock Holmes language model (LinkedIn, 1920 x 1080, 30 fps, silent, captions burned in).

All model numbers on screen come from the kept checkpoint (model_c256_newdata_mix/best.pt): the generated text,
the probabilities, the token IDs, the embeddings, the hidden states, the W_Q products and the attention weights.
The training facts come from MODEL.md. The script checks them with assert before it draws.
The browser window is drawn; its text is real model output.

Run from the project root:
    python portfolio/make_video.py
"""

import glob
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import build_model
from src.utils import setup_console

W, H = 1920, 1080
FPS = 30
OUT = Path(__file__).with_suffix(".mp4")
CKPT = ROOT / "artifacts/checkpoints/model_c256_newdata_mix/best.pt"

BG, PANEL, PANEL2, LINE = (11, 14, 20), (22, 27, 34), (30, 36, 45), (48, 54, 61)
FG, MUTED = (230, 237, 243), (139, 148, 158)
AMBER, BLUE, RED, GREEN = (242, 179, 61), (88, 166, 255), (248, 81, 73), (63, 185, 80)


# ---------------------------------------------------------------- real numbers from the model

def model_facts() -> dict:
    ck = torch.load(CKPT, weights_only=False, map_location="cpu")
    model, tok = build_model(ck["config"], ck["which"])
    model.load_state_dict(ck["model"])
    model.eval()
    f = {"step": ck["step"], "val": ck["val_loss"]}
    f["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert f["params"] == 7_449_856 and round(f["val"], 3) == 3.410, (f["params"], f["val"])

    @torch.no_grad()
    def next_probs(prompt):
        return torch.softmax(model(torch.tensor([tok.encode(prompt)]))[0][0, -1], -1)

    def top(prompt, k=5):
        v, i = next_probs(prompt).topk(k)
        return [(tok.decode([j]), p) for p, j in zip(v.tolist(), i.tolist())]

    # Knowledge cards
    f["cards"] = [(p, top(p, 4)) for p in ["221B Baker", "Professor", "Dr. Watson and Mr. Sherlock"]]
    for (_, t5), e in zip(f["cards"], [" Street", " Mor", " Holmes"]):
        assert t5[0][0] == e, (t5, e)
    ids = tok.encode("Professor")
    with torch.no_grad():
        for _ in range(5):
            ids.append(int(model(torch.tensor([ids]))[0][0, -1].argmax()))
    assert tok.decode(ids).startswith("Professor Moriarty"), tok.decode(ids)
    f["mori_i"] = next_probs("Professor Mor")[tok.encode("Professor Mori")[-1]].item()

    # "Elementary, my dear Watson" is in none of the 9 books, only in the article sets
    for b in ["book", "memoirs", "return", "hound", "lastbow", "signfour", "scarlet", "valley", "casebook"]:
        assert "Elementary, my dear" not in (ROOT / f"data/{b}.txt").read_text(encoding="utf-8"), b
    assert "Elementary, my dear Watson" in (ROOT / "data/articles.txt").read_text(encoding="utf-8")

    # Generation trace for the browser (temperature 0.8, top-k 40, seed 2)
    prompt, end = "“My dear Holmes,” said I,", "no sign of any kind.”"
    torch.manual_seed(2)
    ids = torch.tensor([tok.encode(prompt)])
    n_prompt, steps = ids.shape[1], []
    with torch.no_grad():
        for _ in range(120):
            v, i = (model(ids[:, -256:])[0][0, -1] / 0.8).topk(40)
            p = torch.softmax(v, -1)
            k = torch.multinomial(p, 1).item()
            steps.append({"tok": tok.decode([i[k].item()]), "p": p[k].item(),
                          "top": [(tok.decode([i[j].item()]), p[j].item()) for j in range(5)]})
            ids = torch.cat([ids, i[k].view(1, 1)], 1)
            if tok.decode(ids[0, n_prompt:].tolist()).endswith(end):
                break
    gen = tok.decode(ids[0, n_prompt:].tolist())
    assert gen.endswith(end), gen
    f["prompt"], f["steps"], f["gen"] = prompt, steps, gen
    words = lambda t: re.findall(r"[a-z’']+", t.lower())
    cw = words(" ".join(Path(p).read_text(encoding="utf-8") for p in glob.glob(str(ROOT / "data/*.txt"))
                        if "sources" not in p))
    grams = {tuple(cw[k:k + 8]) for k in range(len(cw) - 7)}
    gw = words(prompt + gen)
    assert not any(tuple(gw[k:k + 8]) in grams for k in range(len(gw) - 7)), "sample copies training text"

    # One forward pass of "“Elementary, my dear", step by step, for the "inside the model" scene
    elem = "“Elementary, my dear"
    eids = tok.encode(elem)
    f["elem"], f["elem_ids"] = elem, eids
    f["elem_toks"] = [tok.decode([i]) for i in eids]
    f["elem_top"] = top(elem)
    assert f["elem_top"][0][0] == " Watson"
    layers = []
    with torch.no_grad():
        x = model.embedding(torch.tensor([eids]))
        f["emb"] = x[0].numpy()
        for b in model.blocks:
            b.attention.inspect = True
            hn = b.norm1(x)
            q = b.attention.W_Q(hn)[0, -1]
            r = int(q.abs().argmax())
            w, h = b.attention.W_Q.weight[r], hn[0, -1]
            assert abs((w * h).sum().item() - q[r].item()) < 1e-4
            x = b(x)
            cum = torch.cumsum(w * h, 0)
            assert abs(cum[-1].item() - q[r].item()) < 1e-4
            layers.append({"r": r, "w": w.tolist(), "h": h.tolist(), "q": q[r].item(), "cum": cum.tolist(),
                           "att": b.attention.saved["weights"][0].mean(0).numpy(), "hid": x[0, -1].numpy()})
        logits = model.output(model.final_norm(x))[0, -1]
        assert torch.allclose(logits, model(torch.tensor([eids]))[0][0, -1], atol=1e-4)
    f["layers"], f["logits"] = layers, logits.numpy()
    f["watson_id"] = tok.encode(elem + " Watson")[-1]
    assert int(logits.argmax()) == f["watson_id"]

    # Attention: the 8 heads of layer index 3 on the story sentence; head index 6 is the large one
    sent = "To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any other name."
    ids = torch.tensor([tok.encode(sent)])
    with torch.no_grad():
        model(ids)
    w = model.blocks[3].attention.saved["weights"][0]
    f["att_tokens"] = [tok.decode([i]) for i in ids[0].tolist()]
    f["att8"] = w.numpy()
    w6 = w[6]
    f["att_prev"] = sum(w6[t, t - 1].item() for t in range(1, w6.shape[0])) / (w6.shape[0] - 1)
    assert f["att_prev"] > 0.4, f["att_prev"]
    return f


# Facts from MODEL.md (the training logs were deleted; these are the recorded values)
TRAIN_TOKENS = 1_220_415 + 261_421 + 196_225 + 320_095 + 235_912
assert TRAIN_TOKENS == 2_234_068
PATH = [("First full run", 3.618), ("Dropout 0.2, 20 epochs", 3.520), ("Gemma vectors (within noise)", 3.513),
        ("Context 256 tokens", 3.499), ("+1 M tokens of new text", 3.410)]
BITS, LZMA = 1.452, 2.125
TRAIN_SECONDS = 759 + 262
RANDOM_LOSS = math.log(4000)
assert round(RANDOM_LOSS, 2) == 8.29


# ---------------------------------------------------------------- drawing helpers

_fonts = {}


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    if (name, size) not in _fonts:
        _fonts[(name, size)] = ImageFont.truetype(f"C:/Windows/Fonts/{name}.ttf", size)
    return _fonts[(name, size)]


def clamp(x):
    return max(0.0, min(1.0, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def mix(c1, c2, a):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * a) for i in range(3))


def heat(v, m):
    """Blue = positive, red = negative, brighter = larger."""
    a = clamp(abs(v) / m) ** 0.7
    return mix(PANEL, BLUE if v > 0 else RED, a)


def num(v, nd=3):
    s = f"{v:.{nd}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s.replace("-", "−")


def label(tok: str) -> str:
    return tok.replace("\n", "↵").strip() or "␣"


def make_background() -> Image.Image:
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    shade = (1 - 0.45 * np.clip(r - 0.3, 0, 1))[..., None]
    arr = (np.array(BG, dtype=np.float32) * shade + np.array([4, 6, 10]) * (1 - r.clip(0, 1))[..., None])
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for x in range(0, W, 48):
        for y in range(0, H, 48):
            d.point((x, y), fill=(30, 36, 46))
    return img


BACKGROUND = None


def appear(d, xy, s, f, color, t, start, dur=0.4, anchor="la"):
    a = ease((t - start) / dur)
    if a > 0:
        d.text((xy[0], xy[1] + (1 - a) * 18), s, font=f, fill=mix(BG, color, a), anchor=anchor)


def title(d, t, s, sub=None):
    appear(d, (90, 70), s, font("seguisb", 58), FG, t, 0.0)
    if sub:
        appear(d, (92, 145), sub, font("segoeui", 30), MUTED, t, 0.15)


def bars(d, x, y, w, items, a=1.0, highlight=0, row=56, lw=200, fs=30):
    fl, fv = font("georgia", fs), font("seguisb", fs - 4)
    for n, (tk, p) in enumerate(items):
        yy = y + n * row
        col = AMBER if n == highlight else BLUE
        d.text((x + lw - 20, yy + 18), label(tk), font=fl, fill=FG if n == highlight else MUTED, anchor="rm")
        d.rounded_rectangle((x + lw, yy + 2, x + lw + w, yy + 34), 6, fill=PANEL2)
        bw = max(4, w * p * a)
        d.rounded_rectangle((x + lw, yy + 2, x + lw + bw, yy + 34), 6, fill=col)
        d.text((x + lw + bw + 12, yy + 18), f"{100 * p * a:.1f}%", font=fv, fill=col, anchor="lm")


def colored_wrap(d, x0, y0, segments, f, maxw, lh):
    """segments = [(text, color)]; wrap on spaces; returns the end (x, y) of the text."""
    chars = [(ch, col) for s, col in segments for ch in s]
    x, y, word = x0, y0, []

    def flush(x, y, word):
        wlen = sum(f.getlength(c) for c, _ in word)
        if x + wlen > x0 + maxw and x > x0:
            x, y = x0, y + lh
            while word and word[0][0] == " ":
                word = word[1:]
        for c, col in word:
            d.text((x, y), c, font=f, fill=col)
            x += f.getlength(c)
        return x, y

    for c in chars:
        if c[0] == " " and word:
            x, y = flush(x, y, word)
            word = []
        word.append(c)
    return flush(x, y, word)


def cursor_arrow(d, x, y, pressed=False):
    pts = [(0, 0), (0, 30), (8, 23), (14, 36), (19, 34), (13, 21), (23, 21)]
    d.polygon([(x + a, y + b) for a, b in pts], fill=AMBER if pressed else (245, 245, 245), outline=(0, 0, 0))


def zoom(img, cx, cy, s):
    if s <= 1.001:
        return img
    w, h = W / s, H / s
    x0 = min(max(cx - w / 2, 0), W - w)
    y0 = min(max(cy - h / 2, 0), H - h)
    return img.resize((W, H), Image.LANCZOS, box=(x0, y0, x0 + w, y0 + h))


def heatmap(d, x0, y0, m, cell, gap=1, reveal=None):
    n = m.shape[0]
    for i in range(n):
        for j in range(n):
            if j > i:
                col = (16, 20, 26)
            elif reveal is not None and i >= reveal:
                col = PANEL
            else:
                col = mix(PANEL, BLUE, float(m[i, j]) ** 0.6)
            d.rectangle((x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell - gap, y0 + (i + 1) * cell - gap), fill=col)


# ---------------------------------------------------------------- scenes: draw(img, d, t, F) -> optional new image

def scene_hook(img, d, t, F):
    def typed(s, f, y, start, cps, color):
        n = int(clamp((t - start) * cps / len(s)) * len(s))
        if n <= 0:
            return
        x = W / 2 - f.getlength(s) / 2
        d.text((x, y), s[:n], font=f, fill=color, anchor="ls")
        if n < len(s) or (t - start) * cps < len(s) + 6:
            if int(t * 3) % 2 == 0:
                cx = x + f.getlength(s[:n]) + 8
                d.rectangle((cx, y - f.size * 0.75, cx + 6, y + 6), fill=AMBER)

    typed("I built a GPT from scratch.", font("seguisb", 104), 470, 0.3, 22, FG)
    typed("Then I made it read Sherlock Holmes.", font("georgiai", 92), 620, 2.0, 24, AMBER)
    appear(d, (W / 2, 760), "Here is what 7.4 million numbers learned.", font("segoeui", 44), MUTED, t, 4.0,
           anchor="mm")


def scene_browser(img, d, t, F):
    # window
    X0, Y0, X1, Y1 = 150, 70, 1770, 1010
    d.rounded_rectangle((X0 + 8, Y0 + 14, X1 + 8, Y1 + 14), 18, fill=(5, 7, 10))
    d.rounded_rectangle((X0, Y0, X1, Y1), 18, fill=(17, 21, 28), outline=LINE, width=2)
    d.rounded_rectangle((X0, Y0, X1, Y0 + 64), 18, fill=PANEL2)
    d.rectangle((X0, Y0 + 40, X1, Y0 + 64), fill=PANEL2)
    for k, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse((X0 + 24 + k * 30, Y0 + 24, X0 + 40 + k * 30, Y0 + 40), fill=c)
    d.rounded_rectangle((X0 + 140, Y0 + 14, X0 + 440, Y0 + 64), 10, fill=(17, 21, 28))
    d.text((X0 + 162, Y0 + 40), "Sherlock SLM · playground", font=font("segoeui", 22), fill=FG, anchor="lm")
    d.rounded_rectangle((X0 + 470, Y0 + 18, X1 - 30, Y0 + 52), 17, fill=(17, 21, 28))
    d.text((X0 + 495, Y0 + 35), "localhost:8765", font=font("segoeui", 21), fill=MUTED, anchor="lm")

    LX, RX = X0 + 50, X0 + 1080            # left pane, right pane
    fin = font("georgia", 34)
    d.text((LX, Y0 + 100), "Prompt", font=font("seguisb", 24), fill=MUTED)
    d.rounded_rectangle((LX, Y0 + 138, RX - 40, Y0 + 218), 12, fill=PANEL, outline=LINE, width=2)
    btn = (LX, Y0 + 240, LX + 220, Y0 + 300)

    phase2 = t >= 11.2
    if not phase2:
        prompt, ts, cps = F["prompt"], 0.3, 13
    else:
        prompt, ts, cps = F["elem"], 11.5, 13
    n = int(clamp((t - ts) * cps / len(prompt)) * len(prompt))
    d.text((LX + 24, Y0 + 178), prompt[:n], font=fin, fill=FG, anchor="lm")
    typing = n < len(prompt)
    if typing and int(t * 4) % 2 == 0:
        cx = LX + 28 + fin.getlength(prompt[:n])
        d.rectangle((cx, Y0 + 158, cx + 3, Y0 + 198), fill=AMBER)

    click = 3.0 if not phase2 else 13.0
    pressed = click <= t < click + 0.2
    d.rounded_rectangle(btn, 12, fill=mix(AMBER, (255, 230, 170), 0.6) if pressed else AMBER)
    d.text(((btn[0] + btn[2]) / 2 - 12, (btn[1] + btn[3]) / 2), "Generate" if not phase2 else "Next word",
           font=font("seguisb", 26), fill=(20, 20, 20), anchor="mm")
    tx, ty = btn[2] - 38, (btn[1] + btn[3]) / 2
    d.polygon([(tx, ty - 9), (tx, ty + 9), (tx + 14, ty)], fill=(20, 20, 20))
    # mouse
    if click - 0.7 <= t <= click + 0.5:
        a = ease((t - (click - 0.7)) / 0.6)
        mx = (RX - 200) + ((btn[0] + 150) - (RX - 200)) * a
        my = (Y0 + 420) + ((btn[1] + 34) - (Y0 + 420)) * a
        cursor_arrow(d, mx, my, pressed)

    # output panel
    OY = Y0 + 330
    d.rounded_rectangle((LX, OY, RX - 40, Y1 - 40), 12, fill=PANEL, outline=LINE, width=2)
    d.text((LX + 24, OY + 18), "Output", font=font("seguisb", 22), fill=MUTED)
    steps = F["steps"]
    side_items, side_title, side_hl = None, "", 0
    if not phase2:
        k = min(len(steps), int(max(0, t - 3.3) / 6.2 * len(steps)) + (1 if t > 3.3 else 0))
        if t > 3.3:
            text = "".join(s["tok"] for s in steps[:k])
            fo = font("georgia", 36)
            ex, ey = colored_wrap(d, LX + 24, OY + 60, [(F["prompt"], AMBER), (text, FG)], fo, RX - LX - 110, 54)
            if k < len(steps) and int(t * 5) % 2 == 0:
                d.rectangle((ex + 4, ey + 6, ex + 8, ey + 42), fill=AMBER)
            cur = steps[max(k - 1, 0)]
            side_items = cur["top"]
            side_hl = next((m for m, (tk, _) in enumerate(cur["top"]) if tk == cur["tok"]), -1)
            side_title = f"token {k} of {len(steps)}"
    else:
        if t >= 13.3:
            fo = font("georgia", 64)
            a = ease((t - 13.3) / 0.4)
            d.text((LX + 40, OY + 150), F["elem"], font=fo, fill=FG)
            wx = LX + 40 + fo.getlength(F["elem"] + " ")
            d.text((wx, OY + 150), "Watson", font=fo, fill=mix(PANEL, AMBER, a))
            d.text((LX + 40, OY + 270), f"next word: “Watson”  ·  {100 * F['elem_top'][0][1]:.1f}%",
                   font=font("seguisb", 30), fill=mix(PANEL, MUTED, a))
            side_items, side_title = F["elem_top"], "next word"
    # side pane
    d.text((RX, Y0 + 100), "Next-token probabilities", font=font("seguisb", 26), fill=FG)
    d.text((RX, Y0 + 138), side_title or "waiting for a prompt…", font=font("segoeui", 22), fill=MUTED)
    if side_items:
        bars(d, RX - 40, Y0 + 200, 300, side_items, 1.0, highlight=side_hl, row=62, lw=170, fs=28)
    if not phase2 and t > 3.3:
        cur = steps[max(k - 1, 0)]
        d.text((RX, Y0 + 530), f"chosen: {label(cur['tok'])}  ({100 * cur['p']:.1f}%)", font=font("seguisb", 26),
               fill=AMBER)
        if side_hl == -1:
            d.text((RX, Y0 + 566), "a lower-ranked token: sampling keeps the text varied", font=font("segoeui", 22),
                   fill=MUTED)
    d.text((RX, Y1 - 120), "raw probabilities (no sampling)" if phase2 else "temperature 0.8 · top-k 40",
           font=font("segoeui", 22), fill=MUTED)
    d.text((RX, Y1 - 88), "model: 7.4 M parameters, CPU", font=font("segoeui", 22), fill=MUTED)

    # zoom into "Watson", then the question
    if phase2 and t > 14.6:
        fo = font("georgia", 64)
        wx = LX + 40 + fo.getlength(F["elem"] + " ") + fo.getlength("Watson") / 2
        img = zoom(img, wx, OY + 185, 1 + 1.0 * ease((t - 14.6) / 1.4))
        d2 = ImageDraw.Draw(img)
        a = ease((t - 15.6) / 0.5)
        if a > 0:
            d2.rounded_rectangle((W / 2 - 520, 820, W / 2 + 520, 940), 20, fill=mix(BG, (8, 10, 14), a))
            d2.text((W / 2, 880), "How did it choose “Watson”?", font=font("seguisb", 60),
                    fill=mix(BG, AMBER, a), anchor="mm")
        return img


def scene_inside(img, d, t, F):
    stages = ["tokens", "embeddings", "6 × (attention + MLP)", "4,000 scores", "softmax"]
    times = [0.0, 2.3, 4.3, 11.5, 13.5]
    st = max(k for k, s in enumerate(times) if t >= s)
    appear(d, (90, 60), "Inside the model: one forward pass", font("seguisb", 50), FG, t, 0.0)
    x = 92
    for k, s in enumerate(stages):
        f = font("seguisb" if k == st else "segoeui", 26)
        col = AMBER if k == st else (MUTED if k < st else LINE)
        d.text((x, 140), s, font=f, fill=col)
        x += f.getlength(s) + 18
        if k < len(stages) - 1:
            d.text((x, 140), "→", font=font("segoeui", 26), fill=LINE)
            x += 44

    toks, ids, emb = F["elem_toks"], F["elem_ids"], F["emb"]
    T = len(toks)
    row = min(64, 430 / T)
    Y = 250
    # 1. tokens
    d.text((90, Y - 46), "token  ·  ID", font=font("segoeui", 22), fill=MUTED)
    for k, (tk, i) in enumerate(zip(toks, ids)):
        a = ease((t - 0.4 - 0.25 * k) / 0.35)
        if a <= 0:
            continue
        y = Y + k * row
        dx = (1 - a) * -40
        d.rounded_rectangle((90 + dx, y, 250 + dx, y + row - 12), 8, fill=PANEL2, outline=mix(BG, AMBER, a), width=2)
        d.text((104 + dx, y + (row - 12) / 2), label(tk), font=font("georgia", 26), fill=mix(BG, FG, a), anchor="lm")
        d.text((330 + dx, y + (row - 12) / 2), str(i), font=font("consola", 24), fill=mix(BG, AMBER, a), anchor="rm")
    # 2. embeddings: one row of 256 numbers per token
    m = float(np.abs(emb).max())
    EX = 370
    if t >= 2.3:
        d.text((EX, Y - 46), f"[{T} × 256] embeddings (real values)", font=font("segoeui", 22), fill=MUTED)
        for k in range(T):
            a = clamp((t - 2.5 - 0.15 * k) / 0.8)
            cells = int(256 * a)
            y = Y + k * row
            for j in range(cells):
                d.rectangle((EX + j * 1.75, y, EX + j * 1.75 + 1, y + row - 12), fill=heat(emb[k, j], m))
    # 4b. explanation, bottom left
    notes = ["Split the text into tokens from my own 4,000-token vocabulary.",
             "Look up a vector for each token: a frozen EmbeddingGemma table, then my learned 512 → 256 projection.",
             "Six layers mix the vectors: attention, then an MLP.",
             "Score every one of the 4,000 tokens.",
             "Turn the scores into probabilities."]
    a = ease((t - times[st]) / 0.4)
    d.text((90, 760), f"{st + 1}", font=font("seguisb", 90), fill=mix(BG, AMBER, a))
    for k, line in enumerate(re.findall(r".{1,36}(?:\s|$)", notes[st])):
        d.text((170, 772 + k * 44), line.strip(), font=font("segoeui", 34), fill=mix(BG, FG, a))

    # 3. six layers
    LX0, CW, GAP = 870, 92, 14
    CH = 330
    layers = F["layers"]
    active = int((t - 4.3) / 1.2) if 4.3 <= t < 11.5 else (-1 if t < 4.3 else 6)
    if t >= 4.0:
        a0 = ease((t - 4.0) / 0.4)
        d.text((LX0, Y - 46), "6 layers  (top: attention weights, bottom: state of “dear”)", font=font("segoeui", 22),
               fill=mix(BG, MUTED, a0))
        for L in range(6):
            x = LX0 + L * (CW + GAP)
            on = L < active or active == 6
            act = L == active
            col = AMBER if act else (BLUE if on else LINE)
            d.rounded_rectangle((x, Y, x + CW, Y + CH), 12, fill=PANEL, outline=mix(BG, col, a0), width=3 if act else 2)
            d.text((x + CW / 2, Y + 24), f"L{L + 1}", font=font("seguisb", 26), fill=mix(BG, col, a0), anchor="mm")
            if on or act:
                att = layers[L]["att"]
                cell = min(9, 76 / T)
                heatmap(d, x + (CW - cell * T) / 2, Y + 56, att / att.max(), cell)
                hid = layers[L]["hid"]
                hm = float(np.abs(hid).max())
                for j in range(256):
                    r_, c_ = divmod(j, 16)
                    d.rectangle((x + 6 + c_ * 5, Y + 160 + r_ * 5, x + 10 + c_ * 5, Y + 164 + r_ * 5),
                                fill=heat(hid[j], hm))
                d.text((x + CW / 2, Y + 262), "256", font=font("consola", 18), fill=MUTED, anchor="mm")
        if 0 <= active < 6:   # pulse down the active layer
            px = LX0 + active * (CW + GAP) + CW / 2
            py = Y + CH * (((t - 4.3) % 1.2) / 1.2)
            d.ellipse((px - 7, py - 7, px + 7, py + 7), fill=AMBER)

    # crunch panel: the real products of one row of W_Q and the real normalized state, with the running sum
    px0, py0, px1, py1 = LX0, 620, 1490, 1030
    if t >= 4.0:
        d.rounded_rectangle((px0, py0, px1, py1), 14, fill=(8, 10, 15), outline=LINE, width=2)
    fc = font("consola", 25)
    if 4.3 <= t < 11.5:
        L = min(active, 5)
        lay = layers[L]
        lt = (t - 4.3) - 1.2 * L
        i = max(1, int(clamp(lt / 1.0) * 256))
        d.text((px0 + 24, py0 + 18), f"layer {L + 1} · query {lay['r']} = row {lay['r']} of W_Q · h",
               font=font("consola", 24), fill=AMBER)
        for k, j in enumerate(range(max(0, i - 6), i)):
            age = i - 1 - j
            col = mix(PANEL, FG, 1 - age / 7)
            w_, h_ = lay["w"][j], lay["h"][j]
            d.text((px0 + 24, py0 + 64 + k * 36), f"{j:>3}  {num(w_, 4):>7} × {num(h_, 4):>7} = {num(w_ * h_, 5):>8}",
                   font=fc, fill=col)
        done = i == 256
        d.text((px0 + 24, py0 + 300), "running sum", font=font("segoeui", 22), fill=MUTED)
        d.text((px0 + 24, py0 + 380), num(lay["cum"][i - 1], 4), font=font("consola", 64),
               fill=AMBER if done else FG, anchor="ls")
        d.text((px1 - 24, py0 + 380), f"{i} / 256 products", font=font("consola", 26), fill=MUTED, anchor="rs")
    elif t >= 11.5:
        d.text((px0 + 24, py0 + 18), "output head [4000 × 256] · h  →  4,000 scores", font=font("consola", 24),
               fill=AMBER)
        if t >= 14.8:
            appear(d, (px0 + 24, py0 + 90), "“Elementary, my dear", font("georgia", 44), FG, t, 14.8)
            appear(d, (px0 + 24, py0 + 160), "Watson", font("georgiai", 96), AMBER, t, 15.0)
            appear(d, (px0 + 24, py0 + 300), f"{100 * F['elem_top'][0][1]:.1f}% · the top choice of 4,000 tokens",
                   font("seguisb", 30), FG, t, 15.2)

    # 5. output: 4,000 logits, then softmax
    OX = 1530
    if t >= 11.5:
        a = ease((t - 11.5) / 0.6)
        lg = F["logits"]
        d.text((OX, Y - 46), "4,000 scores (logits)", font=font("segoeui", 22), fill=mix(BG, MUTED, a))
        d.rounded_rectangle((OX, Y, 1840, Y + 200), 12, fill=PANEL, outline=LINE, width=2)
        lo, hi = float(lg.min()), float(lg.max())
        nb = 290
        bins = lg[: (len(lg) // nb) * nb].reshape(nb, -1).max(1)
        for b in range(int(nb * a)):
            hgt = (bins[b] - lo) / (hi - lo) * 170
            d.line((OX + 10 + b, Y + 190, OX + 10 + b, Y + 190 - hgt), fill=BLUE)
        wb = min(F["watson_id"] // (len(lg) // nb), nb - 1)
        if a >= 1:
            d.line((OX + 10 + wb, Y + 8, OX + 10 + wb, Y + 190), fill=AMBER, width=3)
            d.text((OX + 22 + wb, Y + 18), f"“Watson” {num(float(lg[F['watson_id']]), 2)}", font=font("seguisb", 22),
                   fill=AMBER)
    if t >= 13.5:
        a = ease((t - 13.5) / 1.0)
        d.text((OX, Y + 236), "softmax → probabilities", font=font("segoeui", 22), fill=MUTED)
        bars(d, OX - 40, Y + 276, 110, F["elem_top"], a, row=48, lw=140, fs=24)


def scene_attention(img, d, t, F):
    title(d, t, "Watching it pay attention", "Layer 4: eight heads read the same sentence in eight different ways")
    att8, toks = F["att8"], F["att_tokens"]
    n = len(toks)
    cell = 8.5
    rows = int(clamp((t - 0.4) / 1.8) * n)
    for h in range(8):
        r_, c_ = divmod(h, 4)
        x0, y0 = 90 + c_ * 225, 250 + r_ * 280
        hl = h == 6 and t > 2.6
        d.rounded_rectangle((x0 - 10, y0 - 40, x0 + n * cell + 10, y0 + n * cell + 10), 10, fill=PANEL,
                            outline=AMBER if hl else LINE, width=3 if hl else 1)
        d.text((x0, y0 - 30), f"head {h + 1}", font=font("seguisb" if hl else "segoeui", 20), fill=AMBER if hl else MUTED)
        heatmap(d, x0, y0, att8[h], cell, gap=0, reveal=rows)
    if t > 2.6:
        a = ease((t - 2.6) / 0.6)
        big, x0, y0 = 24, 1290, 300
        m = att8[6]
        k = int(clamp((t - 2.8) / 1.6) * n)
        fl = font("segoeui", 19)
        for i, tk in enumerate(toks):
            d.text((x0 - 10, y0 + i * big + big / 2), label(tk), font=fl, fill=mix(BG, FG if i < k else LINE, a),
                   anchor="rm")
            lab = Image.new("RGBA", (140, 24), (0, 0, 0, 0))
            ImageDraw.Draw(lab).text((0, 12), label(tk), font=fl, fill=MUTED + (int(255 * a),), anchor="lm")
            lab = lab.rotate(60, expand=True)
            img.paste(lab, (int(x0 + i * big - 6), int(y0 - lab.height - 4)), lab)
        heatmap(d, x0, y0, m, big, gap=2, reveal=k)
        d.rectangle((x0 - 4, y0 - 4, x0 + n * big + 2, y0 + n * big + 2), outline=mix(BG, AMBER, a), width=2)
        d.text((x0 + n * big, y0 + n * big + 14), "head 7, enlarged", font=font("seguisb", 22),
               fill=mix(BG, AMBER, a), anchor="ra")
    appear(d, (90, 900), f"Head 7 puts {100 * F['att_prev']:.0f}% of its attention on the previous token.",
           font("seguisb", 36), FG, t, 4.6)
    appear(d, (90, 955), "Dark corner = the future: a token can never look ahead.", font("segoeui", 30), MUTED, t, 5.0)


def scene_cards(img, d, t, F):
    title(d, t, "What it picked up from the text", "Real next-token probabilities; no hand-written rules")
    answers = ["Street", "Moriarty", "Holmes"]
    for k, ((prompt, items), ans) in enumerate(zip(F["cards"], answers)):
        s = 0.4 + 0.5 * k
        a = ease((t - s) / 0.5)
        if a <= 0:
            continue
        x = 90 + k * 590
        y = 240 + (1 - a) * 40
        d.rounded_rectangle((x, y, x + 560, y + 600), 20, fill=PANEL, outline=mix(BG, LINE, a), width=2)
        fp = font("georgia", 40 if len(prompt) < 20 else 34)
        d.text((x + 30, y + 40), prompt, font=fp, fill=mix(BG, FG, a))
        appear(d, (x + 30, y + 100), ans, font("georgiai", 64), AMBER, t, s + 0.6)
        g = ease((t - s - 0.5) / 0.8)
        bars(d, x - 10, y + 230, 220, items, g, row=56, lw=170, fs=26)
        if k == 1:
            appear(d, (x + 30, y + 500), f"then “i” at {100 * F['mori_i']:.1f}%", font("seguisb", 28), MUTED, t, s + 1.4)
    appear(d, (W / 2, 925), "“Elementary, my dear” → “Watson” (71%): a phrase that is in none of the 9 books.",
           font("seguisb", 36), FG, t, 2.6, anchor="mm")
    appear(d, (W / 2, 980), "It occurs only in the Wikipedia articles it read.", font("segoeui", 32), MUTED, t, 2.9,
           anchor="mm")


def scene_results(img, d, t, F):
    title(d, t, "How it improved", "Validation loss on text it never trained on (the last 10% of every source), lower is better")
    gx0, gx1, gy0, gy1 = 160, 1060, 280, 700
    lo, hi = 3.38, 3.65
    d.line((gx0, gy1, gx1, gy1), fill=LINE, width=2)
    for v in (3.4, 3.5, 3.6):
        y = gy1 - (v - lo) / (hi - lo) * (gy1 - gy0)
        d.line((gx0, y, gx1, y), fill=(28, 33, 40), width=1)
        d.text((gx0 - 14, y), f"{v:.1f}", font=font("segoeui", 24), fill=MUTED, anchor="rm")
    pts = []
    for n, (name, v) in enumerate(PATH):
        x = gx0 + 70 + n * (gx1 - gx0 - 140) / (len(PATH) - 1)
        pts.append((x, gy1 - (v - lo) / (hi - lo) * (gy1 - gy0), name, v))
    shown = clamp((t - 0.3) / 2.4) * (len(PATH) - 1)
    for n in range(len(PATH) - 1):
        a = clamp(shown - n)
        if a > 0:
            (xa, ya, _, _), (xb, yb, _, _) = pts[n], pts[n + 1]
            d.line((xa, ya, xa + (xb - xa) * a, ya + (yb - ya) * a), fill=BLUE, width=5)
    for n, (x, y, name, v) in enumerate(pts):
        if shown + 0.001 >= n:
            col = AMBER if n == len(PATH) - 1 else BLUE
            d.ellipse((x - 11, y - 11, x + 11, y + 11), fill=col)
            d.text((x, y - 24), f"{v:.3f}", font=font("seguisb", 28), fill=col, anchor="md")
            words, lines, line = name.split(), [], ""
            for w_ in words:
                if font("segoeui", 22).getlength(line + " " + w_) > 170 and line:
                    lines.append(line)
                    line = w_
                else:
                    line = (line + " " + w_).strip()
            lines.append(line)
            for k, ln in enumerate(lines):
                d.text((x, gy1 + 18 + k * 28), ln, font=font("segoeui", 22), fill=MUTED, anchor="ma")
    cards = [(f"{RANDOM_LOSS:.2f} → {F['val']:.2f}", "loss: random guess → trained"),
             (f"{BITS:.2f} bits/char", f"{100 * (1 - BITS / LZMA):.0f}% fewer than lzma compression ({LZMA:.3f})")]
    for k, (big, sub) in enumerate(cards):
        a = ease((t - 2.8 - 0.4 * k) / 0.5)
        if a <= 0:
            continue
        y = 270 + k * 260
        d.rounded_rectangle((1180, y, 1830, y + 220), 20, fill=PANEL, outline=mix(BG, LINE, a), width=2)
        d.text((1505, y + 90), big, font=font("seguisb", 70), fill=mix(BG, AMBER, a), anchor="mm")
        d.text((1505, y + 170), sub, font=font("segoeui", 28), fill=mix(BG, MUTED, a), anchor="mm")
    appear(d, (W / 2, 960), "New, different text improved every held-out part, books included.",
           font("segoeui", 34), FG, t, 4.0, anchor="mm")


def scene_close(img, d, t, F):
    stats = [(f"{F['params'] / 1e6:.1f} M", "parameters"), (f"{TRAIN_TOKENS / 1e6:.1f} M", "training tokens"),
             (f"{round(TRAIN_SECONDS / 60)} min", "of training on one T4 GPU")]
    for k, (big, sub) in enumerate(stats):
        x = 420 + k * 540
        appear(d, (x, 300), big, font("seguisb", 110), AMBER, t, 0.2 + 0.3 * k, anchor="mm")
        appear(d, (x, 400), sub, font("segoeui", 36), MUTED, t, 0.3 + 0.3 * k, anchor="mm")
    appear(d, (W / 2, 560), "Tokenizer · attention · RoPE · SwiGLU · training loop · Colab pipeline",
           font("segoeui", 38), FG, t, 1.4, anchor="mm")
    appear(d, (W / 2, 620), "built from scratch, to understand how LLMs really work.", font("segoeui", 38), FG, t, 1.6,
           anchor="mm")
    appear(d, (W / 2, 790), "Next: pretrain on 150 M tokens of Victorian books.", font("seguisb", 48), AMBER, t, 2.4,
           anchor="mm")


SCENES = [(scene_hook, 6.0), (scene_browser, 17.0), (scene_inside, 17.0), (scene_attention, 8.0),
          (scene_cards, 7.0), (scene_results, 8.0), (scene_close, 6.5)]


def frame(F, t_total: float, total: float) -> Image.Image:
    acc = 0.0
    for fn, dur in SCENES:
        if t_total < acc + dur or fn is SCENES[-1][0]:
            break
        acc += dur
    t = t_total - acc
    img = BACKGROUND.copy()
    d = ImageDraw.Draw(img)
    new = fn(img, d, t, F)
    if new is not None:
        img, d = new, ImageDraw.Draw(new)
    if fn not in (scene_hook, scene_browser):
        d.text((W - 60, 40), "SHERLOCK SLM", font=font("seguisb", 22), fill=AMBER, anchor="ra")
    d.rectangle((0, H - 6, W * t_total / total, H), fill=AMBER)
    fade = min(clamp(t / 0.3), clamp((dur - t) / 0.3)) if fn is not scene_close else clamp(t / 0.3)
    return Image.blend(BACKGROUND, img, fade) if fade < 1 else img


def main() -> None:
    global BACKGROUND
    setup_console()
    F = model_facts()
    print(f"checks passed: {F['params']:,} parameters, val {F['val']:.3f}, sample {len(F['steps'])} tokens, "
          f"prompt tokens {F['elem_toks']}, previous-token share {F['att_prev']:.3f}")
    BACKGROUND = make_background()
    total = sum(d for _, d in SCENES)
    n = int(total * FPS)
    if len(sys.argv) > 1:   # preview: save PNG frames at the given times (seconds) to the given folder
        out = Path(sys.argv[1])
        for s in sys.argv[2:]:
            frame(F, float(s), total).save(out / f"frame_{s}.png")
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-movflags", "+faststart", str(OUT)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        proc.stdin.write(frame(F, i / FPS, total).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    print(f"saved {OUT} ({total:.1f} s, {n} frames)")


if __name__ == "__main__":
    main()
