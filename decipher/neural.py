"""Small causal transformer LM, its training loop, and scoring helpers.

Used at two levels: over cipher symbols (character-LM segmenter) and over
English words (neural prior for decipherment key search). The architecture
here is a minimal placeholder; swap in your own model as long as it maps
(B, T) token ids to (B, T, vocab) logits.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


class Vocab:
    def __init__(self, items):
        self.itos = [PAD, BOS, EOS] + sorted(set(items))
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.bos, self.eos = 0, 1, 2

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.bos] + [self.stoi[t] for t in toks] + [self.eos]


class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        T = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))


def build_model(vocab_size, cfg):
    return CausalTransformerLM(vocab_size, cfg["d_model"], cfg["n_layers"], cfg["n_heads"],
                               cfg["d_ff"], cfg["max_len"], cfg["dropout"])


def _batch(seqs, max_len, pad, device):
    seqs = [s[:max_len] for s in seqs]
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x.to(device)


def train_lm(model, seqs, cfg, device, seed, tag="lm"):
    """Train on id sequences (each starting with BOS) with next-token loss."""
    g = torch.Generator().manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    for step in range(cfg["steps"]):
        idx = torch.randint(len(seqs), (cfg["batch_size"],), generator=g).tolist()
        x = _batch([seqs[i] for i in idx], model.max_len + 1, 0, device)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), ignore_index=0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if cfg.get("log_every") and (step + 1) % cfg["log_every"] == 0:
            print(f"  [{tag}] step {step + 1}/{cfg['steps']} loss={loss.item():.3f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def sequence_logprobs(model, seqs, device, batch_size):
    """Total log-probability of each id sequence (excluding the leading BOS)."""
    out = []
    for i in range(0, len(seqs), batch_size):
        x = _batch(seqs[i:i + batch_size], model.max_len + 1, 0, device)
        logp = F.log_softmax(model(x[:, :-1]).float(), -1)
        tgt = x[:, 1:]
        lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * (tgt != 0)
        out.extend(lp.sum(1).tolist())
    return out


@torch.no_grad()
def predictive_entropy(model, seqs, device, batch_size):
    """For each sequence, entropy (nats) of p(token_t | prefix) for every target position."""
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        x = _batch(chunk, model.max_len + 1, 0, device)
        logp = F.log_softmax(model(x[:, :-1]).float(), -1)
        ent = -(logp.exp() * logp).sum(-1).cpu()
        for j, s in enumerate(chunk):
            n = min(len(s), model.max_len + 1) - 1
            out.append(ent[j, :n].tolist() + [math.nan] * (len(s) - 1 - n))
    return out
