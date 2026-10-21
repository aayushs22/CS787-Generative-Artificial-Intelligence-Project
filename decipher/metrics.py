"""Segmentation and decipherment metrics."""

from collections import Counter


def _prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _spans(units):
    out, i = [], 0
    for u in units:
        out.append((i, i + len(u)))
        i += len(u)
    return out


def segmentation_metrics(gold, pred):
    """Boundary P/R/F (utterance-internal boundaries), plus Goldwater et al.'s
    word-token P/R/F (both ends correct) and lexicon P/R/F (word types)."""
    b_tp = b_pred = b_gold = t_tp = t_pred = t_gold = 0
    for g, p in zip(gold, pred):
        gs, ps = _spans(g), _spans(p)
        gb, pb = {e for _, e in gs[:-1]}, {e for _, e in ps[:-1]}
        b_tp += len(gb & pb)
        b_pred += len(pb)
        b_gold += len(gb)
        t_tp += len(set(gs) & set(ps))
        t_pred += len(ps)
        t_gold += len(gs)
    g_lex = {u for s in gold for u in s}
    p_lex = {u for s in pred for u in s}
    out = {}
    for name, vals in [("boundary", _prf(b_tp, b_pred, b_gold)),
                       ("token", _prf(t_tp, t_pred, t_gold)),
                       ("lexicon", _prf(len(g_lex & p_lex), len(p_lex), len(g_lex)))]:
        out.update({f"{name}_p": vals[0], f"{name}_r": vals[1], f"{name}_f": vals[2]})
    return out


def _align(ref, hyp):
    """Levenshtein alignment; returns (edits, matches)."""
    n, m = len(ref), len(hyp)
    d = [[(j, 0) for j in range(m + 1)]] + [[(i, 0)] + [None] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = ref[i - 1] == hyp[j - 1]
            sub = (d[i - 1][j - 1][0] + (not same), d[i - 1][j - 1][1] + same)
            dele = (d[i - 1][j][0] + 1, d[i - 1][j][1])
            ins = (d[i][j - 1][0] + 1, d[i][j - 1][1])
            d[i][j] = min(sub, dele, ins, key=lambda x: (x[0], -x[1]))
    return d[n][m]


def decipherment_metrics(plain, decoded, true_key, pred_key, freq_bands):
    """Token accuracy and normalised edit distance after alignment, sentence
    accuracy, and vocabulary recovery per frequency band.

    A word type w counts as recovered if the predicted segmentation contains
    w's true code as a unit and the predicted key maps that unit to w.
    """
    edits = matches = total = exact = 0
    for ref, hyp in zip(plain, decoded):
        e, m = _align(ref, hyp)
        edits += e
        matches += m
        total += len(ref)
        exact += ref == hyp
    out = {"token_acc": matches / total, "norm_edit": edits / total,
           "sentence_acc": exact / len(plain)}

    freq = Counter(w for s in plain for w in s)
    for lo, hi in freq_bands:
        words = [w for w, c in freq.items() if c >= lo and (hi is None or c <= hi)]
        ok = sum(pred_key.get(true_key[w]) == w for w in words)
        name = f"vocab_{lo}-{hi if hi is not None else 'inf'}"
        out[name] = ok / len(words) if words else None
        out[name + "_n"] = len(words)
    ok = sum(pred_key.get(true_key[w]) == w for w in freq)
    out["vocab_all"] = ok / len(freq)
    return out
