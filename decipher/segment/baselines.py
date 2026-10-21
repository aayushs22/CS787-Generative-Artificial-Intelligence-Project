"""Non-Bayesian segmenters: perturbed oracle and branching entropy."""

import math
import random
from collections import Counter, defaultdict

import numpy as np


def perturb(gold, rate, seed):
    """Corrupt a gold segmentation: each token is, with prob `rate`, merged with
    the next token or split at a random interior point (50/50)."""
    rng = random.Random(seed)
    out = []
    for units in gold:
        res, i = [], 0
        while i < len(units):
            u = units[i]
            if rng.random() < rate:
                if rng.random() < 0.5 and i + 1 < len(units):
                    res.append(u + units[i + 1])
                    i += 2
                    continue
                if len(u) > 1:
                    cut = rng.randint(1, len(u) - 1)
                    res.extend([u[:cut], u[cut:]])
                    i += 1
                    continue
            res.append(u)
            i += 1
        out.append(res)
    return out


def scores_to_segments(texts, scores, quantile):
    """Turn per-position boundary scores into segmentations.

    scores[s][i] scores a boundary before character i (i = 1..len-1). A boundary
    is placed at local maxima whose score is at or above the corpus-wide
    `quantile` of all scores.
    """
    flat = np.concatenate([np.asarray(sc[1:], dtype=float) for sc in scores if len(sc) > 1] or [np.zeros(1)])
    thr = float(np.quantile(flat, quantile))
    segs = []
    for s, sc in zip(texts, scores):
        cuts = [0]
        for i in range(1, len(s)):
            left = sc[i - 1] if i > 1 else -math.inf
            right = sc[i + 1] if i + 1 < len(s) else -math.inf
            if sc[i] >= thr and sc[i] >= left and sc[i] >= right:
                cuts.append(i)
        cuts.append(len(s))
        segs.append([s[a:b] for a, b in zip(cuts, cuts[1:])])
    return segs


def _entropy(counter):
    tot = sum(counter.values())
    return -sum(c / tot * math.log(c / tot) for c in counter.values())


def entropy_segment(texts, context, quantile):
    """Branching-entropy segmenter (Harris successor variety, both directions).

    Score before position i = H(next char | k chars to the left)
                            + H(previous char | k chars to the right).
    """
    fwd, bwd = defaultdict(Counter), defaultdict(Counter)
    for s in texts:
        p = "^" * context + s + "$" * context
        for i in range(context, len(p)):
            fwd[p[i - context:i]][p[i]] += 1
            bwd[p[i - context + 1:i + 1]][p[i - context]] += 1
    hf = {k: _entropy(v) for k, v in fwd.items()}
    hb = {k: _entropy(v) for k, v in bwd.items()}

    scores = []
    for s in texts:
        p = "^" * context + s + "$" * context
        sc = [0.0] * len(s)
        for i in range(1, len(s)):
            j = i + context  # index of s[i] inside p
            sc[i] = hf[p[j - context:j]] + hb[p[j:j + context]]
        scores.append(sc)
    return scores_to_segments(texts, scores, quantile)
