"""Embedding-alignment decipherment (proposal 4.2 stretch goal).

Word embeddings are trained independently on the segmented cipher and on English
(PPMI + truncated SVD), then aligned without any dictionary. Because the cipher is an
exact relabelling of English the two spaces are near-isometric. Deviation from the
proposal: initialisation uses the isometry-based sorted-similarity trick of Artetxe et
al. (2018, VecMap) rather than Conneau et al.'s adversarial training; refinement is
Conneau et al.'s iterative Procrustes with CSLS retrieval.
"""

from collections import Counter

import numpy as np

from .solve import _ranked, decode


def embeddings(corpus, types, dim, window):
    """PPMI (context-distribution smoothing 0.75) + SVD, rows L2-normalised."""
    idx = {w: i for i, w in enumerate(types)}
    n = len(types)
    M = np.zeros((n, n))
    for s in corpus:
        ids = [idx[w] for w in s if w in idx]
        for i, a in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    M[a, ids[j]] += 1.0
    rows = M.sum(1, keepdims=True) + 1e-12
    ctx = M.sum(0) ** 0.75
    pmi = np.log(np.maximum(M * ctx.sum() / (rows * ctx[None, :] + 1e-12), 1e-12))
    ppmi = np.maximum(pmi, 0.0)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    X = U[:, :dim] * np.sqrt(S[:dim])
    X -= X.mean(0)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _sorted_sim(X):
    S = np.sort(X @ X.T, axis=1)
    S = np.sqrt(np.maximum(S, 0))  # as in VecMap
    return S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)


def _csls(A, B, k=10):
    """CSLS similarity between rows of A and rows of B (both L2-normalised)."""
    sim = A @ B.T
    ra = np.sort(sim, axis=1)[:, -k:].mean(1)
    rb = np.sort(sim, axis=0)[-k:].mean(0)
    return 2 * sim - ra[:, None] - rb[None, :]


def _procrustes(X, Z, pairs):
    a, b = zip(*pairs)
    U, _, Vt = np.linalg.svd(X[list(a)].T @ Z[list(b)])
    return U @ Vt


def _one_to_one(score):
    from scipy.optimize import linear_sum_assignment

    r, c = linear_sum_assignment(-score)
    return dict(zip(r.tolist(), c.tolist()))


def embed_align(units_corpus, english, cfg, log=print):
    """Return (key, decoded). Aligns the `max_types` most frequent units of each side."""
    n = min(cfg["max_types"], len(_ranked(units_corpus)), len(_ranked(english)))
    cip, eng = _ranked(units_corpus)[:n], _ranked(english)[:n]
    X = embeddings(units_corpus, cip, cfg["dim"], cfg["window"])
    Z = embeddings(english, eng, cfg["dim"], cfg["window"])

    # unsupervised initial dictionary from sorted self-similarity profiles
    m = min(cfg["init_types"], n)
    init = _csls(_sorted_sim(X[:m]), _sorted_sim(Z[:m]))
    pairs = list(_one_to_one(init).items())
    for it in range(cfg["iterations"]):
        W = _procrustes(X, Z, pairs)
        # restrict to the frequent head first, then grow, as in Conneau et al.'s refinement
        head = min(n, cfg["init_types"] + (it + 1) * (n // max(1, cfg["iterations"])))
        asg = _one_to_one(_csls(X[:head] @ W, Z[:head]))
        new = list(asg.items())
        changed = sum(dict(pairs).get(a) != b for a, b in new)
        pairs = new
        log(f"  [embed] iter {it + 1}: {head} types, {changed} pairs changed")
    key = {cip[a]: eng[b] for a, b in pairs}
    return key, decode(units_corpus, key)
