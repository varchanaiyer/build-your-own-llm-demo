# Build Your Own LLM — Interactive Demo

A hands-on web app that walks students through **the entire process of building and
running a language model**, step by step — the companion to the Algoverse
*"Build Your Own LLM"* lecture.

Students don't just read about LLMs; they **select**, **configure**, **train**, and
**run** a real (tiny) GPT in the browser, with live output at every stage.

## The 8 steps (mirrors the lecture)

1. **Data Selection** — choose what the model learns from (Shakespeare, Austen, recipes, code…). *Garbage in = garbage out.*
2. **Tokenization** — type text and watch it become character ids using a vocabulary.
3. **Embeddings** — see cosine similarity turn ids into "meaning vectors" (cat ≈ kitten, cat ≠ car).
4. **Attention** — see Query/Key/Value scores + softmax decide what each word attends to.
5. **Architecture** — pick a GPT-2 preset (small / medium / full) like choosing a car brand.
6. **Hyperparameters** — set batch size, epochs, learning rate, weight decay, context length.
7. **Train** — run the real training loop with a **live loss curve** and the model writing as it learns.
8. **Generate** — give a prompt and sample text with temperature / top-k / top-p.

Each step shows the same explanations as the lecture, plus the *real* output of a
PyTorch model running underneath.

## Quick start (local)

```bash
cd build-your-own-llm-demo
./run.sh
```

Then open **http://127.0.0.1:8000**. First run installs dependencies (~1–2 min);
after that it starts instantly.

### Manual start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## Run with Docker

```bash
docker build -t byollm .
docker run -p 8000:8000 byollm
```

## Deploy

It's a single FastAPI process that serves both the API and the static frontend,
so it deploys anywhere that runs a container or a Python web service:

- **Render / Railway / Fly.io**: point at the `Dockerfile`, expose port `8000`.
- **Any VM**: `./run.sh` behind nginx, or `uvicorn ... --host 0.0.0.0`.
- For a public host, bind `0.0.0.0` and put it behind HTTPS.

> Training runs on CPU and is intentionally tiny (trains in seconds). The model is
> small enough to fit in a few MB — great for teaching, not for production text.

## How it works (for the curious / for instructors)

```
backend/
  model.py      # the tiny char-level GPT — attention, feed-forward, blocks, generate()
  tokenizer.py  # character-level vocab + encode/decode
  engine.py     # training loop (background thread), live status, sampling, demos
  app.py        # FastAPI routes; serves the frontend too
  data/         # public-domain training corpora
frontend/
  index.html    # the 8-step walkthrough
  app.js        # calls the API, draws the live loss curve
  styles.css
```

Every function in `model.py` maps to a slide: `CausalSelfAttention`, `FeedForward`,
`Block`, token/positional embeddings, and `generate()` with temperature/top-k/top-p.

## Extending it (good student experiments)

- Add your own `.txt` to `backend/data/` — it shows up automatically in Step 1.
- Train on `austen` vs `python_code` and compare the generated style.
- Generate at temperature 0.2 vs 1.4 — accurate vs creative.
- Bump the architecture to `gpt2-full` and watch params + training time grow.
