"""Plumbing around the team's decoder-only LMs (decipher/models): vocabulary,
model construction, a generic training loop, and scoring helpers.

Used at two levels: over cipher symbols (character-LM segmenter) and over
English words (the from-scratch LLM used for decipherment, see llm.py).
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import ARCHS

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
IGNORE = -100
_MODEL_KEYS = ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff", "max_seq_len", "dropout",
               "pos_emb_type", "attention_scale", "norm_first")


class Vocab:
    def __init__(self, items):
        self.itos = [PAD, BOS, EOS] + sorted(set(items))
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.bos, self.eos = 0, 1, 2
        self.n_special = 3

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.bos] + [self.stoi[t] for t in toks] + [self.eos]


def build_model(vocab_size, cfg):
    """cfg: {"arch": "decoder1" | "decoder3", <LMConfig fields>}. Missing LMConfig
    fields fall back to the defaults in the team's model file."""
    mod = ARCHS[cfg["arch"]]
    unknown = set(cfg) - set(_MODEL_KEYS) - {"arch"}
    if unknown:
        raise ValueError(f"unknown model config keys: {sorted(unknown)}")
    kw = {k: v for k, v in cfg.items() if k != "arch" and v is not None}
    return mod.DecoderLM(mod.LMConfig(vocab_size=vocab_size, **kw))


class Backbone(nn.Module):
    """Wrapper exposing the attributes the team's technique files expect
    (.lm, .kind, .blocks, .d_model, .max_len). forward returns LM logits."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.kind = "decoder"
        self.d_model = lm.cfg.d_model
        self.max_len = lm.cfg.max_seq_len

    @property
    def blocks(self):
        return list(self.lm.blocks)

    def forward(self, idx):
        return self.lm(idx)


def max_len(model):
    return model.max_len if isinstance(model, Backbone) else model.cfg.max_seq_len


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# --- batching ----------------------------------------------------------------

def pad_batch(seqs, length, pad=0):
    x = torch.full((len(seqs), length), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:length]
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return x


def lm_batch(seqs, limit):
    """Next-token batch from id sequences (each starting with BOS): (inputs, targets)."""
    seqs = [s[:limit + 1] for s in seqs]
    ids = pad_batch(seqs, max(map(len, seqs)))
    x, y = ids[:, :-1], ids[:, 1:].clone()
    y[y == 0] = IGNORE
    return x, y


# --- training ------------------------------------------------------------------

def _autocast(device, amp):
    if amp in (None, "none") or not str(device).startswith("cuda"):
        return torch.autocast("cpu", enabled=False), None
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]
    scaler = torch.amp.GradScaler("cuda") if amp == "fp16" else None
    return torch.autocast("cuda", dtype=dtype), scaler


def _lr_at(step, cfg):
    warm, total = cfg["warmup_steps"], cfg["steps"]
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * t))


def ce_loss(logits, y, x):
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE)


@torch.enable_grad()  # may be first called lazily from inside a no_grad scoring function
def train_lm(model, make_batch, cfg, device, seed, tag="lm", val_batches=None, loss_fn=ce_loss):
    """Generic trainer. make_batch(generator) -> (x, y) with y = IGNORE where no loss.
    loss_fn(logits, y, x) -> scalar (default cross-entropy; KD passes its own).
    Keeps the best-validation weights when val_batches is given. Returns history."""
    g = torch.Generator().manual_seed(seed)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ctx, scaler = _autocast(device, cfg.get("amp"))
    hist = {"train": [], "val": []}
    best, best_state = math.inf, None

    def evaluate():
        model.eval()
        tot = n = 0
        with torch.no_grad(), ctx:
            for x, y in val_batches:
                x, y = x.to(device), y.to(device)
                k = int((y != IGNORE).sum())
                tot += ce_loss(model(x), y, x).item() * k
                n += k
        model.train()
        return tot / max(1, n)

    model.train()
    for step in range(cfg["steps"]):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, cfg)
        x, y = make_batch(g)
        x, y = x.to(device), y.to(device)
        with ctx:
            loss = loss_fn(model(x), y, x)
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        hist["train"].append(loss.item())
        if cfg.get("log_every") and (step + 1) % cfg["log_every"] == 0:
            print(f"  [{tag}] step {step + 1}/{cfg['steps']} loss={loss.item():.3f} lr={_lr_at(step, cfg):.2e}",
                  flush=True)
        last = step + 1 == cfg["steps"]
        if val_batches and ((cfg.get("eval_every") and (step + 1) % cfg["eval_every"] == 0) or last):
            v = evaluate()
            hist["val"].append((step + 1, v))
            print(f"  [{tag}] step {step + 1} val loss={v:.3f} ppl={math.exp(min(v, 50)):.1f}", flush=True)
            if v < best:
                best, best_state = v, copy.deepcopy({k: t.detach().cpu() for k, t in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
        hist["best_val"] = best
    model.eval()
    return hist


# --- scoring -------------------------------------------------------------------

@torch.no_grad()
def sequence_logprobs(model, seqs, device, batch_size):
    """Total log-probability of each id sequence (excluding the leading BOS)."""
    out = []
    for i in range(0, len(seqs), batch_size):
        x, y = lm_batch(seqs[i:i + batch_size], max_len(model))
        x, y = x.to(device), y.to(device)
        logp = F.log_softmax(model(x).float(), -1)
        mask = y != IGNORE
        lp = logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1) * mask
        out.extend(lp.sum(1).tolist())
    return out


@torch.no_grad()
def predictive_entropy(model, seqs, device, batch_size):
    """For each sequence, entropy (nats) of p(token_t | prefix) for every target position."""
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        x, _ = lm_batch(chunk, max_len(model))
        logp = F.log_softmax(model(x.to(device)).float(), -1)
        ent = -(logp.exp() * logp).sum(-1).cpu()
        for j, s in enumerate(chunk):
            n = min(len(s) - 1, x.size(1))
            out.append(ent[j, :n].tolist() + [math.nan] * (len(s) - 1 - n))
    return out
