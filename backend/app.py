"""FastAPI backend for "Build Your Own LLM".

One process serves both the JSON API (under /api) and the static frontend, so
deployment is a single command. Each endpoint maps to a step of the lecture.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import engine

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
FRONTEND = BASE.parent / "frontend"

app = FastAPI(title="Build Your Own LLM")


# ---- dataset catalog -------------------------------------------------------

def _list_datasets() -> list[dict]:
    items = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        items.append({
            "id": path.stem,
            "name": path.stem.replace("_", " ").title(),
            "chars": len(text),
            "words": len(text.split()),
            "preview": text[:240].strip(),
        })
    return items


@app.get("/api/datasets")
def datasets():
    return {"datasets": _list_datasets()}


class SelectData(BaseModel):
    ids: list[str]


@app.post("/api/select-data")
def select_data(body: SelectData):
    """Step 3: concatenate the chosen corpora and build the tokenizer."""
    parts = []
    for ds in body.ids:
        path = DATA_DIR / f"{ds}.txt"
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    if not parts:
        return {"ok": False, "error": "No valid datasets selected."}
    text = "\n\n".join(parts)
    info = engine.build_tokenizer(text)
    return {"ok": True, **info}


# ---- tokenization preview --------------------------------------------------

class TokenizeBody(BaseModel):
    text: str


@app.post("/api/tokenize")
def tokenize(body: TokenizeBody):
    if engine.SESSION.tokenizer is None:
        return {"ok": False, "error": "Select training data first."}
    return {"ok": True, **engine.SESSION.tokenizer.preview(body.text)}


# ---- concept demos ---------------------------------------------------------

@app.get("/api/embeddings-demo")
def embeddings_demo():
    return engine.embedding_demo()


@app.get("/api/attention-demo")
def attention_demo():
    return engine.attention_demo()


@app.get("/api/architectures")
def architectures():
    return {"presets": engine.PRESETS}


# ---- training --------------------------------------------------------------

class TrainBody(BaseModel):
    arch: str = "gpt2-small"
    block_size: int = 128
    batch_size: int = 32
    epochs: int = 10
    max_steps: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    eval_interval: int = 50


@app.post("/api/train")
def train(body: TrainBody):
    cfg = engine.TrainConfig(
        batch_size=body.batch_size,
        epochs=body.epochs,
        max_steps=body.max_steps,
        learning_rate=body.learning_rate,
        weight_decay=body.weight_decay,
        eval_interval=body.eval_interval,
    )
    return engine.start_training(cfg, body.arch, body.block_size)


@app.get("/api/train/status")
def train_status():
    return engine.status()


@app.post("/api/train/stop")
def train_stop():
    return engine.stop_training()


# ---- inference -------------------------------------------------------------

class GenBody(BaseModel):
    prompt: str = "First Citizen:"
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 0.0


@app.post("/api/generate")
def generate(body: GenBody):
    return engine.generate(
        body.prompt, body.max_new_tokens, body.temperature,
        body.top_k or None, body.top_p or None,
    )


# ---- frontend (served last so /api routes win) -----------------------------

@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="static")
