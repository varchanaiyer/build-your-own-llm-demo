"""A tiny, readable char-level GPT — the "brain" from the lecture.

This mirrors the GPT architecture diagram step by step:
    Input  ->  Token Embedding + Positional Encoding
           ->  N x Transformer Block (Self-Attention -> Feed-Forward, each with
                                      a residual connection and LayerNorm)
           ->  Final LayerNorm -> Linear -> (softmax at sampling time)

It is intentionally small so it trains on a laptop CPU in under a minute.
Everything here maps onto a slide from the "Build Your Own LLM" lecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128   # context length: how many tokens it looks back at
    n_layer: int = 3        # number of transformer blocks stacked
    n_head: int = 4         # number of attention heads
    n_embd: int = 128       # embedding dimension (the "meaning vector" size)
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Component 1 from the lecture: the Self-Attention layer.

    Each token produces a Query ("what am I looking for?"), a Key ("what do I
    contain?") and a Value ("what info do I give?"). Tokens compare queries to
    keys to decide who to pay attention to, then mix the values accordingly.
    "Causal" means a token can only look *backwards*, never at future tokens.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # one linear layer produces Q, K and V together (3 * n_embd)
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # lower-triangular mask so position t can only attend to positions <= t
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x, return_attn: bool = False):
        B, T, C = x.shape  # batch, time (tokens), channels (n_embd)
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # reshape into separate heads
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # attention scores = how relevant each token is to each other token
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)  # turn scores into probabilities
        att_weights = att
        att = self.attn_dropout(att)
        y = att @ v  # weighted sum of values
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        if return_attn:
            return y, att_weights
        return y


class FeedForward(nn.Module):
    """Component 2: the Feed-Forward Network. Expand 4x, apply a non-linearity
    (GELU), then shrink back. This is where the model "thinks about what it
    learned" for each token independently."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One Transformer block: Self-Attention then Feed-Forward, each wrapped in
    a residual connection (the `x + ...`) and preceded by Layer Normalization."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection
        x = x + self.ff(self.ln2(x))     # residual connection
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)   # Embeddings
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)   # Positional Encoding
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)         # Output layer
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)  # meaning + position
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            # next-token prediction: compare each prediction to the true next char
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Autoregressive generation: predict next token, append, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]  # keep within context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # temperature scaling

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            if top_p is not None and 0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                # mask tokens once cumulative probability passes top_p (nucleus)
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_probs[remove] = 0.0
                probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    @torch.no_grad()
    def attention_for(self, idx, layer: int = 0, head: int = 0):
        """Return attention weights for one layer/head — used by the demo to
        visualize which tokens attend to which (the "it -> animal" slide)."""
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        attn = None
        for i, block in enumerate(self.blocks):
            if i == layer:
                _, attn = block.attn(block.ln1(x), return_attn=True)
                break
            x = block(x)
        # attn: (B, n_head, T, T) -> pick head
        return attn[0, head]
