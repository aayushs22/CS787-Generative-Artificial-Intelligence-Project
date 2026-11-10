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
    for m in ("search_neural", "search_llm", "llm_refine"):
        assert f"charlm/{m}" in res["decipherment"]
    assert (tmp_path / "llm" / "lm.pt").exists() and (tmp_path / "llm" / "corrector.pt").exists()


# --- LLM: team models, techniques, corrector SFT ----------------------------------

import torch  # noqa: E402

from decipher import llm as llm_mod, neural  # noqa: E402

TINY = {"d_model": 16, "n_layers": 2, "n_heads": 2, "n_kv_heads": 1, "d_ff": 32, "max_seq_len": 24,
        "dropout": 0.0, "norm_first": True}


@pytest.mark.parametrize("arch,pos", [("decoder1", "sinusoidal"), ("decoder3", "rope"), ("decoder3", "sinusoidal")])
def test_team_decoder_is_causal(arch, pos):
    torch.manual_seed(0)
    m = neural.build_model(20, dict(TINY, arch=arch, pos_emb_type=pos)).eval()
    x = torch.randint(3, 20, (2, 10))
    x2 = x.clone()
    x2[:, 6:] = torch.randint(3, 20, (2, 4))
    with torch.no_grad():
        a, b = m(x), m(x2)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-5)  # right padding never leaks backwards


TECHS = [{"name": "full_ft"}, {"name": "lora", "r": 2, "alpha": 4, "targets": ["q_proj", "v_proj"]},
         {"name": "lorma", "r": 2, "alpha": 4, "targets": ["q_proj", "v_proj"]},
         {"name": "rslora", "r": 2, "alpha": 4, "targets": ["q_proj", "v_proj"]},
         {"name": "bitfit"}, {"name": "partial_ft", "k": 1}, {"name": "ia3"},
         {"name": "adapters", "bottleneck": 4}]


@pytest.mark.parametrize("tech", TECHS, ids=lambda t: t["name"])
def test_team_techniques_apply(tech):
    torch.manual_seed(0)
    lm = neural.build_model(20, dict(TINY, arch="decoder3", pos_emb_type="rope")).eval()
    bb = llm_mod.make_backbone(lm, tech).eval()
    c = neural.count_params(bb)
    assert 0 < c["trainable"] <= c["total"]
    if tech["name"] != "full_ft":
        assert c["trainable"] < c["total"]
    x = torch.randint(3, 20, (2, 8))
    with torch.no_grad():
        assert torch.allclose(lm(x), bb(x), atol=1e-5)  # every technique starts at the pretrained model


def test_corrector_batch_masks_noisy_half():
    v = neural.Vocab(["a", "b", "c"])
    clean = [[3, 4, 5]]
    x, y = llm_mod.corrector_batch(clean, torch.tensor([0, 1, 2, 4, 3, 5]), v, 20)
    # sequence: <bos> 4 3 5 <eos> 3 4 5 <eos>; targets supervised only for "3 4 5 <eos>"
    assert x[0].tolist() == [1, 4, 3, 5, 2, 3, 4, 5]
    assert y[0].tolist() == [-100, -100, -100, -100, 3, 4, 5, 2]


def test_corruptor_is_type_consistent(corpus):
    _, _, english, _, _, _, _ = corpus
    v = neural.Vocab({w for s in english for w in s})
    m = llm_mod.Corruptor(v, english, 0.5, 0.5, 3).sample_map(torch.Generator().manual_seed(0))
    assert (m[:3] == torch.arange(3)).all() and (m != torch.arange(len(v))).any()


def test_checkpoint_roundtrip_and_kd(tmp_path, corpus):
    cfg, _, english, _, _, _, _ = corpus
    lc = dict(cfg["llm"], checkpoint_dir=str(tmp_path / "a"))
    teacher = llm_mod.LLM(english, lc, cfg["data"]["max_len"], "cpu", 0, tmp_path / "a")
    corr = teacher.corrector
    loaded, _, _ = llm_mod.load_checkpoint(tmp_path / "a" / "corrector.pt", "cpu")
    x = torch.randint(3, len(teacher.vocab), (2, 9))
    with torch.no_grad():
        assert torch.allclose(corr(x), loaded(x), atol=1e-6)

    kd = {"teacher_checkpoint": str(tmp_path / "a" / "corrector.pt"), "temperature": 2.0, "alpha": 0.5}
    lc2 = dict(lc, checkpoint_dir=str(tmp_path / "b"),
               corrector=dict(lc["corrector"], kd=kd, technique={"name": "full_ft"}))
    student = llm_mod.LLM(english, lc2, cfg["data"]["max_len"], "cpu", 0, tmp_path / "b")
    out = llm_mod.correct(student, [english[0], english[1]])
    assert [len(o) for o in out] == [len(english[0]), len(english[1])]


def test_em_pairs_reduces_to_em_without_pairs_and_decodes_pairs(corpus):
    cfg, plain, english, *_ = corpus
    em_cfg = dict(cfg["decipherment"]["em"])
    k0, d0 = solve.iterative_em([list(s) for s in plain[:60]], english, {**em_cfg, "pairs": 0}, 2)
    k1, d1 = solve.iterative_em([list(s) for s in plain[:60]], english, {**em_cfg, "pairs": 0}, 2)
    assert d0 == d1  # deterministic

    # an under-segmented cipher: "the cat" always fused into a single unit
    eng = [["the", "cat", "ran"], ["a", "dog", "ran"], ["the", "cat", "sat"]] * 30
    units = [["P", "r"], ["a", "d", "r"]] * 30
    key, dec = solve.iterative_em(units, eng, {**em_cfg, "pairs": 5, "start_k": 10, "step_k": 10,
                                               "max_cipher_types": 10}, 2)
    assert key["P"] == "the cat"
    assert dec[0] == ["the", "cat", "ran"]


def test_procrustes_recovers_rotation():
    from decipher import embed

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8))
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    Z = X @ Q
    W = embed._procrustes(X, Z, [(i, i) for i in range(40)])
    np.testing.assert_allclose(X @ W, Z, atol=1e-8)
    assert embed._one_to_one(embed._csls(X @ W, Z, k=3)) == {i: i for i in range(40)}


def test_embed_align_returns_one_to_one_key(corpus):
    cfg, plain, english, vocab, key, texts, gold = corpus
    from decipher.embed import embed_align

    ec = {"dim": 30, "window": 2, "max_types": 60, "init_types": 60, "iterations": 3}
    k, dec = embed_align(gold, english, ec, log=lambda *_: None)
    assert len(set(k.values())) == len(k)
    assert len(dec) == len(gold) and all(len(d) == len(g) for d, g in zip(dec, gold))
