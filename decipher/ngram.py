"""Word n-gram language model with Witten-Bell interpolated smoothing."""

import math
from collections import Counter, defaultdict

import numpy as np

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class NgramLM:
    def __init__(self, sentences, order, vocab=None):
        self.order = order
        self.vocab = sorted(set(vocab) if vocab else {w for s in sentences for w in s})
        self.V = len(self.vocab) + 1  # + EOS
        self.counts = [defaultdict(Counter) for _ in range(order)]  # counts[k][ctx of len k][w]
        for s in sentences:
            toks = [BOS] * (order - 1) + list(s) + [EOS]
            for i in range(order - 1, len(toks)):
                for k in range(order):
                    self.counts[k][tuple(toks[i - k:i])][toks[i]] += 1
        self.totals = [{c: sum(v.values()) for c, v in ck.items()} for ck in self.counts]
        self._cache = {}

    def prob(self, w, ctx):
        ctx = tuple(ctx)[-(self.order - 1):] if self.order > 1 else ()
        key = (w, ctx)
        p = self._cache.get(key)
        if p is None:
            p = self._prob(w, ctx)
            self._cache[key] = p
        return p

    def _prob(self, w, ctx):
        p = 1.0 / self.V
        for k in range(len(ctx) + 1):
            c = ctx[len(ctx) - k:]
            dist = self.counts[k].get(c)
            if not dist:
                continue
            tot, types = self.totals[k][c], len(dist)
            lam = tot / (tot + types)
            p = lam * dist[w] / tot + (1 - lam) * p
        return p

    def sentence_logprob(self, sent):
        toks = [BOS] * (self.order - 1) + list(sent) + [EOS]
        return sum(math.log(self.prob(toks[i], toks[i - self.order + 1:i]))
                   for i in range(self.order - 1, len(toks)))

    def bigram_matrices(self, states):
        """For an HMM over `states`: start[j]=P(j|<s>), trans[i,j]=P(j|i), end[i]=P(</s>|i).
        Uses only the last word of context, so order>2 models are truncated."""
        def ctx(w):
            return ([BOS] * (self.order - 2) + [w]) if self.order > 1 else []

        start = np.array([self.prob(s, ctx(BOS) if self.order > 1 else []) for s in states])
        trans = np.array([[self.prob(t, ctx(s)) for t in states] for s in states])
        end = np.array([self.prob(EOS, ctx(s)) for s in states])
        return start, trans, end
