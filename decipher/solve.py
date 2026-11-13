"""Decipherment: frequency baseline, Iterative EM (Ravi & Knight 2011, §2.1),
and simulated-annealing key search scored by an n-gram or neural English LM."""

import math
import random
from collections import Counter, defaultdict

import numpy as np

from .ngram import UNK, NgramLM


def decode(units_corpus, key):
    return [[key.get(u, UNK) for u in units] for units in units_corpus]


def _ranked(corpus):
    return [w for w, _ in Counter(w for s in corpus for w in s).most_common()]


# --- 1. frequency baseline ---------------------------------------------------

def frequency_key(units_corpus, english):
    eng = _ranked(english)
    return {u: eng[min(i, len(eng) - 1)] for i, u in enumerate(_ranked(units_corpus))}


# --- 2. Iterative EM -----------------------------------------------------------

def _forward_backward(obs, start, trans, end, B):
    """Batched scaled forward-backward. obs: (M, T) int. Returns gamma (M, T, S)."""
    M, T = obs.shape
    S = start.shape[0]
    alpha = np.empty((M, T, S))
    a = start[None, :] * B[:, obs[:, 0]].T
    alpha[:, 0] = a / a.sum(1, keepdims=True)
    for t in range(1, T):
        a = (alpha[:, t - 1] @ trans) * B[:, obs[:, t]].T
        alpha[:, t] = a / np.maximum(a.sum(1, keepdims=True), 1e-300)
    beta = np.empty((M, T, S))
    beta[:, -1] = end[None, :]
    for t in range(T - 2, -1, -1):
        b = (B[:, obs[:, t + 1]].T * beta[:, t + 1]) @ trans.T
        beta[:, t] = b / np.maximum(b.sum(1, keepdims=True), 1e-300)
    g = alpha * beta
    return g / np.maximum(g.sum(2, keepdims=True), 1e-300)


def _viterbi(obs, start, trans, end, B, beam):
    """Viterbi restricted to the `beam` most likely states per observation."""
    logA, logs, loge = np.log(trans + 1e-300), np.log(start + 1e-300), np.log(end + 1e-300)
    logB = np.log(B + 1e-300)
    cands = [np.argsort(-logB[:, o])[:beam] for o in obs]
    delta = logs[cands[0]] + logB[cands[0], obs[0]]
    back = []
    for t in range(1, len(obs)):
        sc = delta[:, None] + logA[np.ix_(cands[t - 1], cands[t])]
        back.append(sc.argmax(0))
        delta = sc.max(0) + logB[cands[t], obs[t]]
    j = int(np.argmax(delta + loge[cands[-1]]))
    path = [j]
    for bp in reversed(back):
        j = int(bp[j])
        path.append(j)
    path.reverse()
    return [int(cands[t][j]) for t, j in enumerate(path)]


def _pair_states(english, words, n_pairs):
    """The `n_pairs` most frequent English bigrams whose words are both in `words`."""
    ws = set(words)
    c = Counter((a, b) for s in english for a, b in zip(s, s[1:]) if a in ws and b in ws)
    return [p for p, _ in c.most_common(n_pairs)]


def _expand(start_w, trans_w, end_w, widx, states):
    """HMM over word states and English word-pair states (a, b). A pair state emits one
    cipher unit but stands for two words, so its transition factors are
    P(first | prev last word) and the internal P(b | a)."""
    first = np.array([widx[s[0] if isinstance(s, tuple) else s] for s in states])
    last = np.array([widx[s[1] if isinstance(s, tuple) else s] for s in states])
    inner = np.array([trans_w[widx[s[0]], widx[s[1]]] if isinstance(s, tuple) else 1.0 for s in states])
    return (start_w[first] * inner, trans_w[np.ix_(last, first)] * inner[None, :], end_w[last])


def _words_of(state):
    return list(state) if isinstance(state, tuple) else [state]


def iterative_em(units_corpus, english, cfg, lm_order, log=print):
    """Grow the channel over the most frequent types; fix confident mappings
    (P(c|e) > fix_threshold) between stages; decode with Viterbi using
    P(e) * P(c|e)^stretch. Returns (key, decoded).

    If cfg["pairs"] = P > 0, the HMM also has P English word-pair states, so a cipher
    unit may map to a word pair (an under-segmented unit). Such keys are returned as
    the string "w1 w2"; decoded sentences are flattened to words."""
    eng_rank = _ranked(english)
    cip_rank = _ranked(units_corpus)[: cfg["max_cipher_types"]]
    uniq = Counter(tuple(s) for s in units_corpus)
    fixed = {}  # english word -> cipher unit

    k = cfg["start_k"]
    while True:
        k_e, k_c = min(k, len(eng_rank)), min(k, len(cip_rank))
        words = eng_rank[:k_e] + [UNK]
        states = words + _pair_states(english, eng_rank[:k_e], cfg.get("pairs", 0))
        e_idx = {w: i for i, w in enumerate(states)}
        cunits = cip_rank[:k_c]
        c_idx = {u: i for i, u in enumerate(cunits)}
        C, S = len(cunits) + 1, len(states)  # last column = unknown cipher unit

        wset = set(words)
        mapped = [[w if w in wset else UNK for w in s] for s in english]
        lm = NgramLM(mapped, lm_order, vocab=words)
        start, trans, end = _expand(*lm.bigram_matrices(words), {w: i for i, w in enumerate(words)}, states)

        B = np.full((S, C), 1.0 / C)
        fixed_rows = {}
        for e, c in fixed.items():
            if e in e_idx and c in c_idx:
                fixed_rows[e_idx[e]] = c_idx[c]
        for r, c in fixed_rows.items():
            B[r] = 0.0
            B[r, c] = 1.0

        by_len = defaultdict(list)
        for s, n in uniq.items():
            by_len[len(s)].append(([c_idx.get(u, C - 1) for u in s], n))

        for it in range(cfg["iterations"]):
            counts = np.zeros((S, C))
            ll = 0.0
            for T, items in by_len.items():
                for i in range(0, len(items), cfg["batch_size"]):
                    chunk = items[i:i + cfg["batch_size"]]
                    obs = np.array([o for o, _ in chunk])
                    w = np.array([n for _, n in chunk], dtype=float)
                    gamma = _forward_backward(obs, start, trans, end, B) * w[:, None, None]
                    for t in range(T):
                        np.add.at(counts.T, obs[:, t], gamma[:, t])
            counts += cfg["smoothing"]
            B_new = counts / counts.sum(1, keepdims=True)
            for r, c in fixed_rows.items():
                B_new[r] = B[r]
            B = B_new
        n_conf = sum(B[r].max() > cfg["fix_threshold"] for r in range(S) if states[r] != UNK)
        log(f"  [em] stage k={k}: {S} plain x {C} cipher types, {n_conf} confident mappings")

        for r in range(S):
            if states[r] == UNK:
                continue
            c = int(B[r].argmax())
            if B[r, c] > cfg["fix_threshold"] and c < C - 1:
                fixed[states[r]] = cunits[c]
        if k_e >= len(eng_rank) and k_c >= len(cip_rank):
            break
        k += cfg["step_k"]

    Bs = B ** cfg["stretch"]
    decoded = []
    votes = defaultdict(Counter)
    for s in units_corpus:
        obs = [c_idx.get(u, C - 1) for u in s]
        path = _viterbi(obs, start, trans, end, Bs, cfg["beam"])
        decoded.append([w for j in path for w in _words_of(states[j])])
        for u, j in zip(s, path):
            votes[u][" ".join(_words_of(states[j]))] += 1
    key = {u: v.most_common(1)[0][0] for u, v in votes.items()}
    return key, decoded


# --- 3. simulated-annealing key search ---------------------------------------

class NgramScorer:
    def __init__(self, lm):
        self.lm = lm

    def __call__(self, sents):
        return np.array([self.lm.sentence_logprob(s) for s in sents])


class NeuralScorer:
    def __init__(self, model, vocab, device, batch_size):
        self.model, self.vocab, self.device, self.bs = model, vocab, device, batch_size

    def __call__(self, sents):
        from .neural import sequence_logprobs

        ids = [self.vocab.encode(s) for s in sents]
        return np.array(sequence_logprobs(self.model, ids, self.device, self.bs))


def key_search(units_corpus, english, init_key, scorer, cfg, seed, log=print, candidates_fn=None):
    """Metropolis search over keys. Each move picks a cipher unit u and a new
    English word w for it: from `candidates_fn` (LLM proposals) with probability
    `candidate_prob`, otherwise near u's frequency rank or uniformly. If another
    unit v already holds w, u and v swap words, so a one-to-one key stays
    one-to-one. Score = sum of LM log-probs of the decoded sentences (a fixed
    subsample of at most `max_sentences` unique cipher sentences).

    candidates_fn(sents, weight, key) -> {unit: [words]} is refreshed every
    `refresh_every` steps."""
    rng = random.Random(seed)
    eng_rank = _ranked(english)
    eng_set = set(eng_rank)
    units = _ranked(units_corpus)
    fallback = frequency_key(units_corpus, english)
    key = {u: (init_key.get(u) if init_key.get(u) in eng_set else fallback[u]) for u in units}
    holders = defaultdict(set)
    for u, w in key.items():
        holders[w].add(u)

    uniq = list(Counter(tuple(s) for s in units_corpus).items())
    rng.shuffle(uniq)
    uniq = uniq[: cfg["max_sentences"]]
    sents = [s for s, _ in uniq]
    weight = np.array([n for _, n in uniq], dtype=float)
    where = defaultdict(set)
    for i, s in enumerate(sents):
        for u in s:
            where[u].add(i)
    units = [u for u in units if u in where]
    unit_w = np.array([math.sqrt(sum(weight[i] for i in where[u])) for u in units])
    unit_cdf = np.cumsum(unit_w / unit_w.sum())
    rank = {u: i for i, u in enumerate(units)}

    cur = scorer([[key[u] for u in s] for s in sents])
    total = float(cur @ weight)
    log(f"  [search] start logprob={total:.1f} over {len(sents)} unique sentences")
    steps = cfg["steps"]
    accepted, cands = 0, None
    for step in range(steps):
        if candidates_fn and step % cfg["refresh_every"] == 0:
            cands = candidates_fn(sents, weight, key)
        temp = cfg["t_start"] + (cfg["t_end"] - cfg["t_start"]) * step / max(1, steps - 1)
        u = units[min(int(np.searchsorted(unit_cdf, rng.random())), len(units) - 1)]
        if cands and rng.random() < cfg["candidate_prob"]:
            new = rng.choice(cands[u])
        elif rng.random() < 0.5:
            r = min(rank[u], len(eng_rank) - 1)
            lo, hi = max(0, r - cfg["rank_window"]), min(len(eng_rank) - 1, r + cfg["rank_window"])
            new = eng_rank[rng.randint(lo, hi)]
        else:
            new = rng.choice(eng_rank)
        if new == key[u]:
            continue
        change = {u: new}
        others = holders[new] - {u}
        if others:
            change[rng.choice(sorted(others))] = key[u]
        affected = sorted(set().union(*(where[x] for x in change)))
        old = {x: key[x] for x in change}
        key.update(change)
        new_lp = scorer([[key[x] for x in sents[i]] for i in affected])
        delta = float((new_lp - cur[affected]) @ weight[affected])
        if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-9)):
            cur[affected] = new_lp
            total += delta
            accepted += 1
            for x, w in old.items():
                holders[w].discard(x)
            for x, w in change.items():
                holders[w].add(x)
        else:
            key.update(old)
        if cfg.get("log_every") and (step + 1) % cfg["log_every"] == 0:
            log(f"  [search] step {step + 1}/{steps} temp={temp:.3f} logprob={total:.1f} accepted={accepted}")
    return key, decode(units_corpus, key)
