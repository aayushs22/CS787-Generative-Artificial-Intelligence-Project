"""From-scratch LLM for decipherment.

Stage 1 (pretrain): the team's DecoderLM trained as a word-level LM on the
    English split only.
Stage 2 (corrector SFT): the pretrained LM is fine-tuned, with one of the
    team's techniques (decipher/techniques) and optionally the team's KD loss,
    to map a corrupted English sentence to the clean one:
        <bos> noisy words <eos> clean words <eos>      (loss on the clean part)
    Corruption imitates a partly wrong key: a random fraction of word types is
    consistently replaced by words of similar frequency rank. Training data is
    English only; no cipher text or key is ever seen.

Decipherment methods:
    search_llm  - key search whose proposals come from the LM's next-word
                  distribution aggregated over every occurrence of a cipher unit,
                  scored by the LM's sentence log-probability.
    llm_refine  - decode with the current key, let the corrector rewrite each
                  sentence, give every cipher unit the word it was most often
                  corrected to (one-to-one, greedy by votes), keep the key only if
                  the LM score improves; repeat.

    python -m decipher.llm --config configs/x.yaml   # train + save both stages only
"""

import argparse
import copy
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from . import data, neural, solve, techniques
from .neural import IGNORE, Backbone, Vocab


# --- pretraining -------------------------------------------------------------

def _split(seqs, frac, seed):
    idx = list(range(len(seqs)))
    random.Random(seed).shuffle(idx)
    n_val = int(len(seqs) * frac)
    return [seqs[i] for i in idx[n_val:]], [seqs[i] for i in idx[:n_val]]


def _sampler(train, batch_size, build):
    def make(g):
        idx = torch.randint(len(train), (batch_size,), generator=g).tolist()
        return build([train[i] for i in idx], g)
    return make


def pretrain(english, cfg, device, seed):
    vocab = Vocab({w for s in english for w in s})
    model = neural.build_model(len(vocab), cfg["model"])
    tc = cfg["pretrain"]
    train, val = _split([vocab.encode(s) for s in english], tc["val_fraction"], seed)
    limit = neural.max_len(model)
    make = _sampler(train, tc["batch_size"], lambda b, g: neural.lm_batch(b, limit))
    val_b = [neural.lm_batch(val[i:i + tc["batch_size"]], limit) for i in range(0, len(val), tc["batch_size"])]
    print(f"  [llm] pretraining {cfg['model']['arch']}: {neural.count_params(model)['total']:,} params, "
          f"vocab {len(vocab)}, {len(train)} train / {len(val)} val sentences", flush=True)
    hist = neural.train_lm(model, make, tc, device, seed, "llm-pretrain", val_b)
    return model, vocab, hist


# --- corrector SFT -------------------------------------------------------------

class Corruptor:
    """Samples consistent word-type substitutions between words of similar frequency."""

    def __init__(self, vocab, english, rate_min, rate_max, rank_window):
        counts = Counter(w for s in english for w in s)
        self.ranked = torch.tensor([vocab.stoi[w] for w, _ in counts.most_common()])
        self.V = len(vocab)
        self.rate_min, self.rate_max, self.window = rate_min, rate_max, rank_window

    def sample_map(self, g):
        """LongTensor m of size V: token id i is replaced by m[i]."""
        m = torch.arange(self.V)
        n = len(self.ranked)
        rate = self.rate_min + (self.rate_max - self.rate_min) * torch.rand(1, generator=g).item()
        chosen = torch.nonzero(torch.rand(n, generator=g) < rate).flatten()
        off = torch.randint(1, self.window + 1, (len(chosen),), generator=g)
        sign = torch.where(torch.rand(len(chosen), generator=g) < 0.5, 1, -1)
        m[self.ranked[chosen]] = self.ranked[(chosen + off * sign).clamp(0, n - 1)]
        return m


def corrector_example(noisy, clean, vocab):
    """Token ids of '<bos> noisy <eos> clean <eos>' and the index where supervision starts."""
    return [vocab.bos] + noisy + [vocab.eos] + clean + [vocab.eos], len(noisy) + 1


def corrector_batch(clean_seqs, mapping, vocab, limit):
    """clean_seqs: word-id lists (no specials). Returns (x, y) with loss only on the clean half."""
    seqs, starts = [], []
    for c in clean_seqs:
        noisy = mapping[torch.tensor(c)].tolist() if mapping is not None else list(c)
        ids, start = corrector_example(noisy, list(c), vocab)
        seqs.append(ids[:limit + 1])
        starts.append(start)
    x, y = neural.lm_batch(seqs, limit)
    for i, st in enumerate(starts):
        y[i, :st] = IGNORE
    return x, y


def _check_len(model, data_max_len):
    need = 2 * data_max_len + 2
    if neural.max_len(model) < need:
        raise ValueError(f"model max_seq_len {neural.max_len(model)} < {need} needed by the corrector "
                         f"(2 * data.max_len + 2)")


def make_backbone(lm, technique):
    bb = Backbone(copy.deepcopy(lm))
    for p in bb.parameters():
        p.requires_grad_(True)
    opts = {k: v for k, v in technique.items() if k != "name"}
    techniques.get(technique["name"]).apply(bb, **opts)
    return bb


def finetune_corrector(lm, vocab, english, cfg, data_max_len, device, seed):
    cc = cfg["corrector"]
    bb = make_backbone(lm, cc["technique"])
    _check_len(bb, data_max_len)
    pc = neural.count_params(bb)
    print(f"  [llm] corrector SFT with {cc['technique']['name']}: {pc['trainable']:,} / {pc['total']:,} "
          f"trainable params", flush=True)
    clean = [[vocab.stoi[w] for w in s] for s in english]
    train, val = _split(clean, cc["val_fraction"], seed + 1)
    corr = Corruptor(vocab, english, cc["corrupt_min"], cc["corrupt_max"], cc["rank_window"])
    limit = neural.max_len(bb)
    make = _sampler(train, cc["batch_size"], lambda b, g: corrector_batch(b, corr.sample_map(g), vocab, limit))
    gv = torch.Generator().manual_seed(seed + 2)
    val_b = [corrector_batch(val[i:i + cc["batch_size"]], corr.sample_map(gv), vocab, limit)
             for i in range(0, len(val), cc["batch_size"])]

    loss_fn = neural.ce_loss
    if cc.get("kd"):
        from .techniques.kd import kd_loss

        kd = dict(cc["kd"])
        teacher, t_vocab, _ = load_checkpoint(kd.pop("teacher_checkpoint"), device)
        if t_vocab.itos != vocab.itos:
            raise ValueError("KD teacher was trained with a different vocabulary")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        def loss_fn(logits, y, x):
            with torch.no_grad():
                t_logits = teacher(x).float()
            return kd_loss(logits.float(), t_logits, y, **kd)

    hist = neural.train_lm(bb, make, cc, device, seed, "llm-corrector", val_b, loss_fn)
    return bb, hist


# --- checkpoints -----------------------------------------------------------------

def save_checkpoint(path, model, vocab, model_cfg, technique=None, hist=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vocab": vocab.itos, "model_cfg": model_cfg, "technique": technique,
                "state": model.state_dict(), "hist": hist}, path)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vocab = Vocab([])
    vocab.itos = ck["vocab"]
    vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
    model = neural.build_model(len(vocab), ck["model_cfg"])
    if ck["technique"] is not None:
        model = make_backbone(model, ck["technique"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), vocab, ck


class LLM:
    """Pretrained LM + corrector, trained on first use or loaded from checkpoint_dir."""

    def __init__(self, english, cfg, data_max_len, device, seed, ckpt_dir):
        self.english, self.cfg, self.device, self.seed = english, cfg, device, seed
        self.data_max_len = data_max_len
        self.dir = Path(ckpt_dir)
        self._lm = self._corrector = None
        self.vocab = None

    def _ckpt(self, name):
        p = self.dir / f"{name}.pt"
        return p if self.cfg.get("reuse_checkpoints") and p.exists() else None

    def _check_vocab(self, vocab):
        want = Vocab({w for s in self.english for w in s}).itos
        if vocab.itos != want:
            raise ValueError("checkpoint vocabulary differs from this run's English split; "
                             "use the same data config or a fresh checkpoint_dir")

    @property
    def lm(self):
        if self._lm is None:
            p = self._ckpt("lm")
            if p:
                self._lm, self.vocab, _ = load_checkpoint(p, self.device)
                self._check_vocab(self.vocab)
                print(f"  [llm] loaded {p}", flush=True)
            else:
                self._lm, self.vocab, hist = pretrain(self.english, self.cfg, self.device, self.seed)
                save_checkpoint(self.dir / "lm.pt", self._lm, self.vocab, self.cfg["model"], None, hist)
        return self._lm

    @property
    def corrector(self):
        if self._corrector is None:
            lm = self.lm
            p = self._ckpt("corrector")
            if p:
                self._corrector, vocab, _ = load_checkpoint(p, self.device)
                self._check_vocab(vocab)
                print(f"  [llm] loaded {p}", flush=True)
            else:
                self._corrector, hist = finetune_corrector(lm, self.vocab, self.english, self.cfg,
                                                           self.data_max_len, self.device, self.seed)
                save_checkpoint(self.dir / "corrector.pt", self._corrector, self.vocab, self.cfg["model"],
                                self.cfg["corrector"]["technique"], hist)
        return self._corrector

    def scorer(self):
        return solve.NeuralScorer(self.lm, self.vocab, self.device, self.cfg["score_batch_size"])


# --- decipherment with the LLM -------------------------------------------------------

def lm_candidates(llm, top_k):
    """Returns fn(sents, weight, key) -> {unit: [top_k English words]}, ranking words by
    the sum over the unit's occurrences of log P(word | decoded left context)."""
    model, vocab, device = llm.lm, llm.vocab, llm.device
    bs = llm.cfg["score_batch_size"]

    @torch.no_grad()
    def fn(sents, weight, key):
        units = sorted({u for s in sents for u in s})
        u_idx = {u: i for i, u in enumerate(units)}
        agg = torch.zeros(len(units), len(vocab), device=device)
        for i in range(0, len(sents), bs):
            chunk = sents[i:i + bs]
            x, _ = neural.lm_batch([vocab.encode([key[u] for u in s]) for s in chunk], neural.max_len(model))
            logp = F.log_softmax(model(x.to(device)).float(), -1)  # position t predicts word t
            rows, pos, uids, w = [], [], [], []
            for j, s in enumerate(chunk):
                for t, u in enumerate(s[:logp.size(1)]):
                    rows.append(j)
                    pos.append(t)
                    uids.append(u_idx[u])
                    w.append(weight[i + j])
            vals = logp[torch.tensor(rows), torch.tensor(pos)] * torch.tensor(w, dtype=torch.float32, device=device)[:, None]
            agg.index_add_(0, torch.tensor(uids, device=device), vals)
        agg[:, :vocab.n_special] = -float("inf")
        top = agg.topk(min(top_k, len(vocab) - vocab.n_special), dim=1).indices.cpu().tolist()
        return {u: [vocab.itos[j] for j in top[u_idx[u]]] for u in units}

    return fn


@torch.no_grad()
def correct(llm, sents):
    """Greedy corrector output for each decoded sentence (same length as the input)."""
    model, vocab, device = llm.corrector, llm.vocab, llm.device
    bs = llm.cfg["score_batch_size"]
    out = [None] * len(sents)
    by_len = defaultdict(list)
    for i, s in enumerate(sents):
        by_len[len(s)].append(i)
    for n, idx in by_len.items():
        for b in range(0, len(idx), bs):
            ids = idx[b:b + bs]
            x = torch.tensor([[vocab.bos] + [vocab.stoi[w] for w in sents[i]] + [vocab.eos] for i in ids],
                             device=device)
            for _ in range(n):
                logits = model(x[:, -neural.max_len(model):])[:, -1].float()
                logits[:, :vocab.n_special] = -float("inf")
                x = torch.cat([x, logits.argmax(-1, keepdim=True)], 1)
            for row, i in zip(x[:, n + 2:].tolist(), ids):
                out[i] = [vocab.itos[j] for j in row]
    return out


def llm_refine(units_corpus, english, init_key, llm, cfg, seed, log=print):
    rng = random.Random(seed)
    eng_set = {w for s in english for w in s}
    fallback = solve.frequency_key(units_corpus, english)
    key = {u: (init_key.get(u) if init_key.get(u) in eng_set else fallback[u]) for u in fallback}
    uniq = list(Counter(tuple(s) for s in units_corpus).items())
    rng.shuffle(uniq)
    uniq = uniq[:cfg["max_sentences"]]
    sents = [s for s, _ in uniq]
    weight = np.array([n for _, n in uniq], dtype=float)
    scorer = llm.scorer()

    def score(k):
        return float(scorer([[k[u] for u in s] for s in sents]) @ weight)

    best = score(key)
    log(f"  [refine] start logprob={best:.1f} over {len(sents)} unique sentences")
    for r in range(cfg["rounds"]):
        fixed = correct(llm, [[key[u] for u in s] for s in sents])
        votes = defaultdict(Counter)
        for s, c, wt in zip(sents, fixed, weight):
            for u, w in zip(s, c):
                votes[u][w] += wt
        props = sorted(((cnt / sum(v.values()), cnt, u, w) for u, v in votes.items() for w, cnt in v.items()),
                       reverse=True)
        new, taken = {}, set()
        for share, _, u, w in props:
            if u not in new and w not in taken and share >= cfg["min_share"]:
                new[u] = w
                taken.add(w)
        cand = {u: new.get(u, key[u]) for u in key}
        changed = sum(cand[u] != key[u] for u in key)
        s_new = score(cand)
        log(f"  [refine] round {r + 1}: {changed} units changed, logprob {best:.1f} -> {s_new:.1f}")
        if s_new <= best or not changed:
            break
        key, best = cand, s_new
    return key, solve.decode(units_corpus, key)


# --- CLI: train both stages without running the pipeline -------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    from .run import _device, _seed_all

    _seed_all(cfg["seed"])
    _, english, _ = data.build_splits(cfg["data"], cfg["seed"])
    out = Path(args.output_dir or cfg["output_dir"])
    lc = cfg["llm"]
    llm = LLM(english, lc, cfg["data"]["max_len"], _device(cfg.get("device", "auto")), cfg["seed"],
              lc.get("checkpoint_dir") or out / "llm")
    llm.corrector  # trains (or loads) the LM, then the corrector
    print(f"saved to {llm.dir}")


if __name__ == "__main__":
    main()
