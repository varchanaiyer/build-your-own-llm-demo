"""Training + generation engine.

Holds a single in-memory "session" so the step-by-step UI can:
  1. build a tokenizer from the chosen data,
  2. train a model in a background thread while reporting live loss,
  3. run inference (text generation) afterwards.

Kept deliberately simple: one global session, guarded by a lock. Perfect for a
classroom demo where one model is being built at a time.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import torch

from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer

# Architecture presets — "different car brands, all transformers" (the slide).
PRESETS = {
    "gpt2-small": {"label": "GPT-2 (small)", "n_layer": 3, "n_head": 4, "n_embd": 128},
    "gpt2-medium": {"label": "GPT-2 (medium)", "n_layer": 4, "n_head": 8, "n_embd": 192},
    "gpt2-full": {"label": "GPT-2 (full-ish)", "n_layer": 6, "n_head": 8, "n_embd": 256},
}


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 10
    max_steps: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    eval_interval: int = 50


@dataclass
class Session:
    tokenizer: CharTokenizer | None = None
    model: GPT | None = None
    data: torch.Tensor | None = None
    arch: str = "gpt2-small"
    block_size: int = 128
    # live training state
    status: str = "idle"  # idle | training | done | error
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    history: list = field(default_factory=list)  # [{step, loss}]
    sample: str = ""
    message: str = ""
    started_at: float = 0.0
    elapsed: float = 0.0
    num_params: int = 0


SESSION = Session()
_LOCK = threading.Lock()
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_tokenizer(text: str) -> dict:
    """Step: Tokenization. Build vocab + encode the whole corpus into a tensor."""
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    with _LOCK:
        SESSION.tokenizer = tok
        SESSION.data = ids
        SESSION.model = None
        SESSION.status = "idle"
        SESSION.history = []
        SESSION.step = 0
    return {
        "vocab_size": tok.vocab_size,
        "num_tokens": int(ids.numel()),
        "vocab_sample": tok.chars[:50],
    }


def _get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(_DEVICE), y.to(_DEVICE)


def _train_loop(cfg: TrainConfig, arch: str, block_size: int):
    try:
        tok = SESSION.tokenizer
        data = SESSION.data
        preset = PRESETS[arch]
        # keep block_size sane for tiny corpora
        block_size = min(block_size, max(16, len(data) // 4))

        model_cfg = GPTConfig(
            vocab_size=tok.vocab_size,
            block_size=block_size,
            n_layer=preset["n_layer"],
            n_head=preset["n_head"],
            n_embd=preset["n_embd"],
        )
        model = GPT(model_cfg).to(_DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        # total training steps = steps/epoch x epochs (the lecture's two knobs)
        total_steps = max(1, cfg.max_steps * cfg.epochs)

        with _LOCK:
            SESSION.model = model
            SESSION.arch = arch
            SESSION.block_size = block_size
            SESSION.status = "training"
            SESSION.total_steps = total_steps
            SESSION.history = []
            SESSION.started_at = time.time()
            SESSION.num_params = model.num_params()

        model.train()
        for step in range(1, total_steps + 1):
            if SESSION.status == "stopped":
                break
            x, y = _get_batch(data, block_size, cfg.batch_size)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 5 == 0 or step == 1:
                with _LOCK:
                    SESSION.step = step
                    SESSION.loss = float(loss.item())
                    SESSION.history.append({"step": step, "loss": float(loss.item())})
                    SESSION.elapsed = time.time() - SESSION.started_at

            if step % cfg.eval_interval == 0 or step == total_steps:
                sample = _sample(model, tok, "First Citizen:", 120, 0.8, block_size)
                with _LOCK:
                    SESSION.sample = sample

        with _LOCK:
            SESSION.status = "done"
            SESSION.step = total_steps
            SESSION.elapsed = time.time() - SESSION.started_at
            SESSION.message = "Training complete."
    except Exception as exc:  # surface errors to the UI instead of dying silently
        with _LOCK:
            SESSION.status = "error"
            SESSION.message = f"{type(exc).__name__}: {exc}"


def start_training(cfg: TrainConfig, arch: str, block_size: int) -> dict:
    if SESSION.tokenizer is None:
        return {"ok": False, "error": "Pick training data first."}
    if SESSION.status == "training":
        return {"ok": False, "error": "Already training."}
    if arch not in PRESETS:
        arch = "gpt2-small"
    t = threading.Thread(target=_train_loop, args=(cfg, arch, block_size), daemon=True)
    t.start()
    return {"ok": True}


def stop_training():
    with _LOCK:
        if SESSION.status == "training":
            SESSION.status = "stopped"
    return {"ok": True}


def status() -> dict:
    with _LOCK:
        return {
            "status": SESSION.status,
            "step": SESSION.step,
            "total_steps": SESSION.total_steps,
            "loss": round(SESSION.loss, 4),
            "history": SESSION.history[-200:],
            "sample": SESSION.sample,
            "message": SESSION.message,
            "elapsed": round(SESSION.elapsed, 1),
            "num_params": SESSION.num_params,
            "arch": SESSION.arch,
        }


def _sample(model, tok, prompt, max_new_tokens, temperature, block_size,
            top_k=None, top_p=None):
    model.eval()
    start = tok.encode(prompt) or [0]
    idx = torch.tensor([start], dtype=torch.long, device=_DEVICE)
    out = model.generate(
        idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p
    )
    model.train()
    return tok.decode(out[0].tolist())


def generate(prompt: str, max_new_tokens: int, temperature: float,
             top_k: int | None, top_p: float | None) -> dict:
    if SESSION.model is None:
        return {"ok": False, "error": "Train a model first."}
    text = _sample(
        SESSION.model, SESSION.tokenizer, prompt or " ", max_new_tokens,
        temperature, SESSION.block_size, top_k or None, top_p or None,
    )
    return {"ok": True, "text": text}


# ---- small interactive demos used by the early lecture steps ---------------

def embedding_demo() -> dict:
    """Recreates the cosine-similarity slide with the lecture's exact vectors."""
    import math

    embeddings = {
        "cat": [0.8, 0.6, -0.2, 0.1],
        "dog": [0.7, 0.5, -0.1, 0.2],
        "kitten": [0.85, 0.65, -0.25, 0.05],
        "car": [-0.5, 0.1, 0.9, 0.3],
        "truck": [-0.4, 0.2, 0.85, 0.4],
    }

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        ma = math.sqrt(sum(x * x for x in a))
        mb = math.sqrt(sum(x * x for x in b))
        return dot / (ma * mb)

    words = list(embeddings)
    matrix = [[round(cos(embeddings[a], embeddings[b]), 3) for b in words] for a in words]
    return {"words": words, "embeddings": embeddings, "similarity": matrix}


def attention_demo() -> dict:
    """Recreates the 'attention weights for sat' slide with a softmax."""
    import math

    scores = {"The": 0.5, "cat": 2.1, "sat": 1.0}
    exps = {k: math.exp(v) for k, v in scores.items()}
    total = sum(exps.values())
    weights = {k: round(v / total, 3) for k, v in exps.items()}
    return {"scores": scores, "weights": weights, "focus": "sat"}
