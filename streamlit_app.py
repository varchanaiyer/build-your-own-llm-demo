"""Build Your Own LLM — Streamlit edition.

Same 8-step walkthrough as the web app, but as a single Streamlit script so it can
be deployed for free on Streamlit Community Cloud. It reuses the *exact* model and
tokenizer code from backend/ — only the UI layer changes.

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub, then https://share.streamlit.io -> New app
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import streamlit as st
import torch

from backend.model import GPT, GPTConfig
from backend.tokenizer import CharTokenizer
from backend.engine import PRESETS, embedding_demo, attention_demo

DATA_DIR = Path(__file__).resolve().parent / "backend" / "data"
DEVICE = "cpu"

st.set_page_config(page_title="Build Your Own LLM", page_icon="◧", layout="wide")

# ---- session state ---------------------------------------------------------
ss = st.session_state
ss.setdefault("tokenizer", None)
ss.setdefault("train_text", "")
ss.setdefault("model", None)
ss.setdefault("block_size", 128)
ss.setdefault("arch", "gpt2-small")
ss.setdefault("trained", False)

# ---- sidebar navigation ----------------------------------------------------
st.sidebar.title("◧ Build Your Own LLM")
st.sidebar.caption("Algoverse AI Research · interactive walkthrough")
STEPS = [
    "1 · Data Selection",
    "2 · Tokenization",
    "3 · Embeddings",
    "4 · Attention",
    "5 · Architecture",
    "6 · Hyperparameters",
    "7 · Train",
    "8 · Generate",
]
step = st.sidebar.radio("Steps", STEPS, label_visibility="collapsed")

vocab_ok = ss.tokenizer is not None
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Vocab built:** {'✅' if vocab_ok else '⬜'}  \n"
    f"**Model trained:** {'✅' if ss.trained else '⬜'}"
)


def list_datasets():
    items = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        items.append({
            "id": path.stem,
            "name": path.stem.replace("_", " ").title(),
            "chars": len(text),
            "words": len(text.split()),
            "preview": text[:200].strip(),
            "text": text,
        })
    return items


# ===========================================================================
# STEP 1 — DATA
# ===========================================================================
if step == STEPS[0]:
    st.header("1 · Pick Your Training Text")
    st.write(
        "A model learns patterns from *whatever you feed it*. Train on Shakespeare and "
        "it writes like Shakespeare; train on recipes and it writes recipes. "
        "**Garbage in = garbage out.**"
    )
    datasets = list_datasets()
    labels = {f"{d['name']}  ({d['words']:,} words)": d for d in datasets}
    picked = st.multiselect("Select one or more corpora:", list(labels.keys()),
                            default=[list(labels.keys())[3]] if len(labels) > 3 else None)
    for key in picked:
        d = labels[key]
        with st.expander(f"Preview · {d['name']}"):
            st.code(d["preview"])

    if st.button("Use this data →", type="primary", disabled=not picked):
        text = "\n\n".join(labels[k]["text"] for k in picked)
        tok = CharTokenizer(text)
        ss.tokenizer = tok
        ss.train_text = text
        ss.model = None
        ss.trained = False
        st.success(
            f"✅ Built a vocabulary of **{tok.vocab_size} unique characters** "
            f"from **{len(text):,} characters** of text."
        )

# ===========================================================================
# STEP 2 — TOKENIZATION
# ===========================================================================
elif step == STEPS[1]:
    st.header("2 · Tokenization")
    st.write(
        "Computers don't understand letters — they understand numbers. We break text into "
        "**tokens** (here, single characters) and map each to an id using a *vocabulary*."
    )
    st.code('"hello" → [\'h\',\'e\',\'l\',\'l\',\'o\'] → [3, 2, 4, 4, 5]')
    if not vocab_ok:
        st.warning("Select & build training data in Step 1 first.")
    else:
        text = st.text_input("Type something to tokenize:", "To be, or not to be")
        prev = ss.tokenizer.preview(text)
        cols = st.columns(min(len(prev["tokens"]), 16) or 1)
        for i, t in enumerate(prev["tokens"][:16]):
            ch = "␣" if t["char"] == " " else t["char"]
            cols[i % len(cols)].metric(ch, t["id"])
        st.markdown(f"**Vocabulary size:** `{prev['vocab_size']}`")
        st.markdown(f"**Encoded:** `{prev['encoded']}`")

# ===========================================================================
# STEP 3 — EMBEDDINGS
# ===========================================================================
elif step == STEPS[2]:
    st.header("3 · Embeddings")
    st.write(
        "An id like `5` doesn't capture meaning. Each token becomes a **vector** — a list of "
        "numbers placing it in 'meaning space'. Similar words end up **close together**."
    )
    d = embedding_demo()
    st.caption("Cosine similarity (1.0 = identical meaning, negative = opposite):")
    import pandas as pd
    df = pd.DataFrame(d["similarity"], index=d["words"], columns=d["words"])
    st.dataframe(df.style.background_gradient(cmap="Oranges", vmin=-1, vmax=1).format("{:.3f}"))
    st.info("cat / dog / kitten score ~0.99 with each other, but cat vs car is negative — "
            "different meaning, far apart.")

# ===========================================================================
# STEP 4 — ATTENTION
# ===========================================================================
elif step == STEPS[3]:
    st.header("4 · The Attention Mechanism")
    st.write(
        "\"The animal didn't cross the street because **it** was too tired.\" What does \"it\" "
        "refer to? **Attention** lets each word look back and decide which others matter."
    )
    st.caption("For each word: **Query** (what am I looking for?), **Key** (what do I contain?), "
               "**Value** (what info do I give?).")
    d = attention_demo()
    st.write(f"Raw relevance scores for `\"{d['focus']}\"`: "
             + ", ".join(f"{k}={v}" for k, v in d["scores"].items()) + " → after softmax:")
    import pandas as pd
    wdf = pd.DataFrame({"word": list(d["weights"].keys()),
                        "attention": list(d["weights"].values())}).set_index("word")
    st.bar_chart(wdf)
    st.info('The model learns that "sat" should pay most attention to "cat" — that\'s who did '
            'the sitting.')

# ===========================================================================
# STEP 5 — ARCHITECTURE
# ===========================================================================
elif step == STEPS[4]:
    st.header("5 · Choose Your Model Style")
    st.write("Architectures are like car brands — all transformers, just different design "
             "choices. Bigger = more capable but slower to train.")
    cols = st.columns(len(PRESETS))
    for col, (pid, p) in zip(cols, PRESETS.items()):
        with col:
            st.markdown(f"### {p['label']}")
            st.caption(f"{p['n_layer']} layers · {p['n_head']} heads · {p['n_embd']}-dim")
    ss.arch = st.radio("Selected architecture:",
                       list(PRESETS.keys()),
                       format_func=lambda x: PRESETS[x]["label"],
                       index=list(PRESETS).index(ss.arch))

# ===========================================================================
# STEP 6 — HYPERPARAMETERS
# ===========================================================================
elif step == STEPS[5]:
    st.header("6 · Hyperparameters")
    st.write("These knobs control *how* the model learns. The defaults train in seconds.")
    c1, c2, c3 = st.columns(3)
    ss.hp_batch = c1.number_input("Batch size", 1, 256, 32, help="Examples learned at once.")
    ss.hp_epochs = c2.number_input("Epochs", 1, 50, 3, help="Passes over the data.")
    ss.hp_steps = c3.number_input("Steps / epoch", 20, 2000, 200)
    ss.hp_lr = c1.number_input("Learning rate", 0.0001, 0.1, 0.001, format="%.4f",
                               help="Step size: too high = chaotic, too low = slow.")
    ss.hp_wd = c2.number_input("Weight decay", 0.0, 0.5, 0.01, format="%.3f",
                               help="Fights memorization.")
    ss.block_size = c3.number_input("Context length", 16, 256, 128,
                                    help="How many characters back the model sees.")
    st.info(f"Total training steps = **{ss.hp_steps} × {ss.hp_epochs} = "
            f"{ss.hp_steps * ss.hp_epochs:,}** weight updates.")

# ===========================================================================
# STEP 7 — TRAIN
# ===========================================================================
elif step == STEPS[6]:
    st.header("7 · Train Your LLM")
    st.write("The loop: show the model text, it predicts the next character, we measure how "
             "wrong it was (the **loss**), and nudge its weights. Repeat — loss should fall.")
    if not vocab_ok:
        st.warning("Select & build training data in Step 1 first.")
    else:
        total_steps = ss.get("hp_steps", 200) * ss.get("hp_epochs", 3)
        if st.button("▶ Start training", type="primary"):
            tok = ss.tokenizer
            data = torch.tensor(tok.encode(ss.train_text), dtype=torch.long)
            block_size = min(ss.block_size, max(16, len(data) // 4))
            preset = PRESETS[ss.arch]
            cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=block_size,
                            n_layer=preset["n_layer"], n_head=preset["n_head"],
                            n_embd=preset["n_embd"])
            model = GPT(cfg).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(),
                                    lr=ss.get("hp_lr", 1e-3),
                                    weight_decay=ss.get("hp_wd", 0.01))
            st.caption(f"Model: {model.num_params():,} parameters · context {block_size}")

            prog = st.progress(0.0)
            metric = st.empty()
            chart = st.line_chart(x_label="step", y_label="loss")
            sample_box = st.empty()
            bs = ss.get("hp_batch", 32)

            model.train()
            start = time.time()
            for stp in range(1, total_steps + 1):
                ix = torch.randint(len(data) - block_size - 1, (bs,))
                x = torch.stack([data[i:i + block_size] for i in ix]).to(DEVICE)
                y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(DEVICE)
                _, loss = model(x, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if stp % 5 == 0 or stp == total_steps:
                    prog.progress(stp / total_steps)
                    metric.markdown(f"Step **{stp}/{total_steps}** · loss **{loss.item():.4f}** "
                                    f"· {time.time() - start:.1f}s")
                    chart.add_rows({"loss": [loss.item()]})
                if stp % 50 == 0 or stp == total_steps:
                    model.eval()
                    idx = torch.tensor([tok.encode("First Citizen:") or [0]], device=DEVICE)
                    out = model.generate(idx, 120, temperature=0.8)
                    sample_box.code(tok.decode(out[0].tolist()))
                    model.train()

            ss.model = model
            ss.trained = True
            st.success(f"✅ Training complete in {time.time() - start:.1f}s — head to Step 8!")

# ===========================================================================
# STEP 8 — GENERATE
# ===========================================================================
elif step == STEPS[7]:
    st.header("8 · Generate Text")
    st.write("Inference time. Give a prompt and the model predicts one token at a time, feeding "
             "each prediction back in. The sampling knobs control how creative it gets.")
    if not ss.trained or ss.model is None:
        st.warning("Train a model first (Step 7).")
    else:
        prompt = st.text_area("Prompt", "First Citizen:", height=70)
        c1, c2, c3, c4 = st.columns(4)
        max_new = c1.number_input("Max new tokens", 10, 1000, 300)
        temp = c2.slider("Temperature", 0.1, 1.5, 0.8, 0.1, help="Randomness / creativity.")
        top_k = c3.number_input("Top-K (0 = off)", 0, 100, 0)
        top_p = c4.number_input("Top-P (0 = off)", 0.0, 1.0, 0.0, 0.05)
        if st.button("Generate", type="primary"):
            tok = ss.tokenizer
            idx = torch.tensor([tok.encode(prompt) or [0]], device=DEVICE)
            with st.spinner("Generating…"):
                out = ss.model.generate(idx, int(max_new), temperature=temp,
                                        top_k=int(top_k) or None, top_p=float(top_p) or None)
            st.code(tok.decode(out[0].tolist()))
        st.info("Try temperature 0.2 (predictable) vs 1.4 (creative but weird). Retrain on a "
                "different dataset in Step 1 to change the style.")
