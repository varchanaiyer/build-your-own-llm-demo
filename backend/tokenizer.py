"""Character-level tokenizer — the simplest kind, exactly as the lecture builds.

    "hello"  ->  ['h','e','l','l','o']  ->  [3, 2, 4, 4, 5]

We build a vocabulary (every unique character), then a two-way mapping between
characters and integer ids. Real models like ChatGPT use sub-word (BPE) tokens,
but characters are the clearest way to *see* what tokenization does.
"""

from __future__ import annotations


class CharTokenizer:
    def __init__(self, text: str):
        # the vocabulary: sorted unique characters in the training text
        chars = sorted(list(set(text)))
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # char  -> index
        self.itos = {i: ch for i, ch in enumerate(chars)}  # index -> char

    def encode(self, s: str) -> list[int]:
        # unknown chars are skipped so demo input can't crash training-derived vocab
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)

    def preview(self, s: str, limit: int = 60) -> dict:
        """Return a token-by-token breakdown for the UI."""
        tokens = []
        for c in s[:limit]:
            if c in self.stoi:
                tokens.append({"char": c, "id": self.stoi[c]})
        return {
            "vocab_size": self.vocab_size,
            "tokens": tokens,
            "encoded": [t["id"] for t in tokens],
            "vocab_sample": self.chars[:40],
        }
