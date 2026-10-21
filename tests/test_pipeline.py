import itertools
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from decipher import cipher, data, metrics, solve
from decipher.ngram import NgramLM
from decipher.segment import GibbsSegmenter, perturb
from decipher.segment.gibbs import END, START

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus():
    cfg = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text())
    plain, english, vocab = data.build_splits(cfg["data"], 0)
    key = cipher.make_key(vocab, cfg["cipher"], 0)
    texts, gold = cipher.encipher(plain, key)
    return cfg, plain, english, vocab, key, texts, gold


def test_splits_are_disjoint_and_in_vocab(corpus):
    _, plain, english, vocab, key, texts, gold = corpus
    vs = set(vocab)
    assert all(w in vs for s in plain + english for w in s)
    assert len(set(key.values())) == len(key)
    assert ["".join(g) for g in gold] == texts


def test_forward_backward_matches_brute_force():
    rng = np.random.default_rng(0)
    S, C, T = 3, 4, 4
    start = rng.dirichlet(np.ones(S))
    trans = rng.dirichlet(np.ones(S), size=S) * 0.8
    end = 1 - trans.sum(1)
    B = rng.dirichlet(np.ones(C), size=S)
    obs = np.array([[0, 2, 1, 3]])
    gamma = solve._forward_backward(obs, start, trans, end, B)[0]

    post = np.zeros((T, S))
    for path in itertools.product(range(S), repeat=T):
        p = start[path[0]] * B[path[0], obs[0, 0]]
        for t in range(1, T):
            p *= trans[path[t - 1], path[t]] * B[path[t], obs[0, t]]
        p *= end[path[-1]]
        for t, s in enumerate(path):
            post[t, s] += p
    np.testing.assert_allclose(gamma, post / post.sum(1, keepdims=True), atol=1e-10)


@pytest.mark.parametrize("model", ["unigram", "bigram"])
def test_gibbs_counts_stay_consistent(corpus, model):
    _, _, _, _, _, texts, _ = corpus
    seg = GibbsSegmenter(model, 8, 20, 10, 0.5, 2, 3, 3, 2, 0.5, seed=1)
    out = seg.fit(texts[:50])
    assert ["".join(u) for u in out] == texts[:50]
    if model == "unigram":
        assert seg.n + Counter() == Counter(w for s in out for w in s)
        assert seg.N == sum(map(len, out))
        assert seg.n_final == len(out)
    else:
        bigrams = Counter((u, w) for s in out for u, w in zip([START] + s, s + [END]))
        assert seg.nb + Counter() == bigrams
        exp_t = Counter()
        for (u, w), c in bigrams.items():
            exp_t[w] += 10 * math.log((c + 10) / 10)
        for w in exp_t:
            assert seg.t[w] == pytest.approx(exp_t[w])
        assert seg.T == pytest.approx(sum(exp_t.values()))


def test_em_solves_cipher_with_matching_lm(corpus):
    """With an LM trained on the plaintext itself, EM should recover most of the key."""
    cfg, plain, _, _, key, _, gold = corpus
    em_cfg = dict(cfg["decipherment"]["em"], iterations=15)
    pred_key, decoded = solve.iterative_em(gold, plain, em_cfg, 2, log=lambda *_: None)
    res = metrics.decipherment_metrics(plain, decoded, key, pred_key, [(1, None)])
    assert res["token_acc"] > 0.8


def test_segmentation_metrics():
    gold = [["AB", "C", "DE"]]
    assert metrics.segmentation_metrics(gold, gold)["token_f"] == 1.0
    r = metrics.segmentation_metrics(gold, [["ABC", "DE"]])
    assert r["boundary_p"] == 1.0 and r["boundary_r"] == 0.5
    assert r["token_p"] == 0.5 and r["token_r"] == pytest.approx(1 / 3)


def test_decipherment_metrics_alignment():
    plain = [["the", "cat", "ran"]]
    key = {"the": "A", "cat": "B", "ran": "C"}
    r = metrics.decipherment_metrics(plain, [["the", "ran"]], key, {"A": "the", "C": "ran"}, [(1, None)])
    assert r["token_acc"] == pytest.approx(2 / 3)
    assert r["norm_edit"] == pytest.approx(1 / 3)
    assert r["vocab_all"] == pytest.approx(2 / 3)


def test_ngram_normalises():
    lm = NgramLM([["a", "b"], ["b", "a", "a"]], 3)
    for ctx in [(), ("a",), ("<s>", "<s>"), ("b", "a"), ("z", "z")]:
        assert sum(lm.prob(w, ctx) for w in ["a", "b", "</s>"]) == pytest.approx(1.0)


def test_perturb_preserves_text(corpus):
    _, _, _, _, _, texts, gold = corpus
    assert ["".join(u) for u in perturb(gold, 0.5, 0)] == texts


def test_end_to_end(tmp_path):
    from decipher.run import main

    res = main(["--config", str(ROOT / "configs/smoke.yaml"), "--output-dir", str(tmp_path)])
    assert (tmp_path / "results.json").exists()
    assert res["segmentation"]["oracle"]["token_f"] == 1.0
    assert "charlm/search_neural" in res["decipherment"]
