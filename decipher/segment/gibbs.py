"""Bayesian word segmentation by Gibbs sampling (Goldwater et al., 2006).

model="unigram": Dirichlet process unigram model (Section 3).
model="bigram":  hierarchical DP bigram model (Section 4 and Appendix), with
                 table counts replaced by their expected values.

The utterance-boundary terms and the indicator corrections in the bigram
denominators follow the corrected equations of Goldwater, Griffiths & Johnson
(2009, Cognition); the 2006 appendix has typos there.
"""

import math
import random
from collections import Counter

START, END = "<s>", "</s>"


class GibbsSegmenter:
    def __init__(self, model, alphabet_size, alpha0, alpha1, p_stop, tau,
                 iterations, anneal_start_temp, anneal_steps, init_boundary_prob, seed,
                 log_every=0):
        if model not in ("unigram", "bigram"):
            raise ValueError(model)
        self.model = model
        self.a0, self.a1, self.tau = alpha0, alpha1, tau
        self.iterations = iterations
        self.t0, self.anneal_steps = anneal_start_temp, anneal_steps
        self.init_p = init_boundary_prob
        self.rng = random.Random(seed)
        self.log_every = log_every
        self._log_p_stop = math.log(p_stop)
        self._log_p_cont = math.log1p(-p_stop)
        self._log_char = -math.log(alphabet_size)
        self._p0_cache = {}

    # --- base distribution -------------------------------------------------
    def p0(self, w):
        p = self._p0_cache.get(w)
        if p is None:
            n = len(w)
            p = math.exp(self._log_p_stop + (n - 1) * self._log_p_cont + n * self._log_char)
            self._p0_cache[w] = p
        return p

    # --- state helpers ------------------------------------------------------
    @staticmethod
    def _words(s, b):
        cuts = [i for i, x in enumerate(b) if x]
        return [s[a:c] for a, c in zip(cuts, cuts[1:])]

    @staticmethod
    def _prev(b, i):
        i -= 1
        while not b[i]:
            i -= 1
        return i

    @staticmethod
    def _next(b, i):
        i += 1
        while not b[i]:
            i += 1
        return i

    def _temperature(self, it):
        anneal_iters = max(1, int(self.iterations * 0.9))
        if it >= anneal_iters or self.anneal_steps <= 0:
            return 1.0
        step = it * self.anneal_steps // anneal_iters
        return self.t0 + (1.0 - self.t0) * step / self.anneal_steps

    # --- unigram counts -----------------------------------------------------
    def _uni_change(self, w, final, d):
        self.n[w] += d
        self.N += d
        if final:
            self.n_final += d

    def _uni_pw(self, w, extra=0, n_extra=0):
        return (self.n[w] + extra + self.a0 * self.p0(w)) / (self.N + n_extra + self.a0)

    # --- bigram counts ------------------------------------------------------
    def _bi_change(self, u, w, d):
        c = self.nb[(u, w)]
        self.nb[(u, w)] = c + d
        self.nc[u] += d
        dt = self.a1 * (math.log((c + d + self.a1) / self.a1) - math.log((c + self.a1) / self.a1))
        self.t[w] += dt
        self.T += dt

    def _p1(self, w):
        t_end = self.t[END]
        if w == END:
            return (t_end + self.tau / 2) / (self.T + self.tau)
        t_sig = self.T - t_end
        return ((t_sig + self.tau / 2) / (self.T + self.tau)
                * (self.t[w] + self.a0 * self.p0(w)) / (t_sig + self.a0))

    def _p2(self, w, u, num=0, den=0):
        return (self.nb[(u, w)] + num + self.a1 * self._p1(w)) / (self.nc[u] + den + self.a1)

    # --- main loop ----------------------------------------------------------
    def fit(self, texts):
        rng = self.rng
        self.texts = texts
        self.b = []
        for s in texts:
            b = bytearray(len(s) + 1)
            b[0] = b[-1] = 1
            for i in range(1, len(s)):
                b[i] = rng.random() < self.init_p
            self.b.append(b)

        if self.model == "unigram":
            self.n, self.N, self.n_final = Counter(), 0, 0
            for s, b in zip(texts, self.b):
                ws = self._words(s, b)
                for j, w in enumerate(ws):
                    self._uni_change(w, j == len(ws) - 1, 1)
        else:
            self.nb, self.nc, self.t, self.T = Counter(), Counter(), Counter(), 0.0
            for s, b in zip(texts, self.b):
                ws = [START] + self._words(s, b) + [END]
                for u, w in zip(ws, ws[1:]):
                    self._bi_change(u, w, 1)

        sample = self._sample_unigram if self.model == "unigram" else self._sample_bigram
        order = [(k, i) for k, s in enumerate(texts) for i in range(1, len(s))]
        for it in range(self.iterations):
            temp = self._temperature(it)
            rng.shuffle(order)
            for k, i in order:
                sample(k, i, temp)
            if self.log_every and (it + 1) % self.log_every == 0:
                print(f"  [{self.model}] iter {it + 1}/{self.iterations} temp={temp:.2f}", flush=True)
        return self.segments()

    def segments(self):
        return [self._words(s, b) for s, b in zip(self.texts, self.b)]

    def _choose(self, p1, p2, temp):
        if temp != 1.0:
            p1, p2 = p1 ** (1 / temp), p2 ** (1 / temp)
        return self.rng.random() * (p1 + p2) < p2  # True -> boundary

    def _sample_unigram(self, k, i, temp):
        s, b = self.texts[k], self.b[k]
        l, r = self._prev(b, i), self._next(b, i)
        w1, w2, w3 = s[l:r], s[l:i], s[i:r]
        final = r == len(s)
        if b[i]:
            self._uni_change(w2, False, -1)
            self._uni_change(w3, final, -1)
        else:
            self._uni_change(w1, final, -1)

        N, nf, tau = self.N, self.n_final, self.tau
        p1 = self._uni_pw(w1) * ((nf if final else N - nf) + tau / 2) / (N + tau)
        p2 = (self._uni_pw(w2) * (N - nf + tau / 2) / (N + tau)
              * self._uni_pw(w3, int(w2 == w3), 1)
              * ((nf if final else N - nf + 1) + tau / 2) / (N + 1 + tau))

        if self._choose(p1, p2, temp):
            b[i] = 1
            self._uni_change(w2, False, 1)
            self._uni_change(w3, final, 1)
        else:
            b[i] = 0
            self._uni_change(w1, final, 1)

    def _sample_bigram(self, k, i, temp):
        s, b = self.texts[k], self.b[k]
        l, r = self._prev(b, i), self._next(b, i)
        w1, w2, w3 = s[l:r], s[l:i], s[i:r]
        wl = START if l == 0 else s[self._prev(b, l):l]
        wr = END if r == len(s) else s[r:self._next(b, r)]
        if b[i]:
            self._bi_change(wl, w2, -1)
            self._bi_change(w2, w3, -1)
            self._bi_change(w3, wr, -1)
        else:
            self._bi_change(wl, w1, -1)
            self._bi_change(w1, wr, -1)

        p1 = (self._p2(w1, wl)
              * self._p2(wr, w1, int(wl == w1 == wr), int(wl == w1)))
        p2 = (self._p2(w2, wl)
              * self._p2(w3, w2, int(wl == w2 == w3), int(wl == w2))
              * self._p2(wr, w3,
                         int(wl == w3 and w2 == wr) + int(w2 == w3 == wr),
                         int(wl == w3) + int(w2 == w3)))

        if self._choose(p1, p2, temp):
            b[i] = 1
            self._bi_change(wl, w2, 1)
            self._bi_change(w2, w3, 1)
            self._bi_change(w3, wr, 1)
        else:
            b[i] = 0
            self._bi_change(wl, w1, 1)
            self._bi_change(w1, wr, 1)
