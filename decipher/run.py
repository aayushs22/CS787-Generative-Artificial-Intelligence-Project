"""End-to-end pipeline: data -> cipher -> segmentation -> decipherment -> metrics.

    python -m decipher.run --config configs/smoke.yaml
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import yaml

from . import cipher, data, metrics, solve
from .ngram import NgramLM
from .segment import GibbsSegmenter, entropy_segment, perturb


def _device(name):
    if name != "auto":
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _write_lines(path, rows):
    path.write_text("\n".join(" ".join(r) if isinstance(r, list) else r for r in rows) + "\n")


def segment(method, texts, gold, cfg, alphabet_size, device, seed):
    if method == "oracle":
        return gold
    if method == "perturbed":
        return perturb(gold, cfg["perturbed"]["rate"], seed)
    if method == "entropy":
        c = cfg["entropy"]
        return entropy_segment(texts, c["context"], c["quantile"])
    if method in ("dp_unigram", "hdp_bigram"):
        c = cfg[method]
        model = "unigram" if method == "dp_unigram" else "bigram"
        return GibbsSegmenter(model, alphabet_size, c["alpha0"], c.get("alpha1", 0.0), c["p_stop"],
                              c["tau"], c["iterations"], c["anneal_start_temp"], c["anneal_steps"],
                              c["init_boundary_prob"], seed, c.get("log_every", 0)).fit(texts)
    if method == "charlm":
        from .segment.charlm import charlm_segment

        alphabet = sorted({ch for s in texts for ch in s})
        return charlm_segment(texts, alphabet, cfg["charlm"], device, seed)
    raise ValueError(f"unknown segmentation method: {method}")


class Decipherer:
    """Runs decipherment methods, building each English LM once."""

    def __init__(self, english, cfg, llm_cfg, data_max_len, device, seed, llm_dir):
        self.english, self.cfg, self.device, self.seed = english, cfg, device, seed
        self._ngram = {}
        self._llm_args = (llm_cfg, data_max_len, llm_dir)
        self._llm = None

    def ngram(self, order):
        if order not in self._ngram:
            self._ngram[order] = NgramLM(self.english, order)
        return self._ngram[order]

    @property
    def llm(self):
        if self._llm is None:
            from .llm import LLM

            llm_cfg, data_max_len, llm_dir = self._llm_args
            self._llm = LLM(self.english, llm_cfg, data_max_len, self.device, self.seed,
                            llm_cfg.get("checkpoint_dir") or llm_dir)
        return self._llm

    def _init(self, units, cache, init):
        if init not in cache:
            cache[init] = self.run(init, units, cache)
        return cache[init][0]

    def run(self, method, units, cache):
        cfg = self.cfg
        if method == "freq":
            key = solve.frequency_key(units, self.english)
            return key, solve.decode(units, key)
        if method == "em_pairs":  # EM whose HMM also has English word-pair states (proposal 4.3)
            return solve.iterative_em(units, self.english, {**cfg["em"], "pairs": cfg["em"].get("pairs", 50)},
                                      cfg["em"]["lm_order"])
        if method == "embed_align":  # unsupervised embedding alignment (proposal 4.2 stretch)
            from .embed import embed_align

            return embed_align(units, self.english, cfg["embed"])
        m = re.fullmatch(r"em(\d)?", method)
        if m:
            order = int(m.group(1) or cfg["em"]["lm_order"])
            if order > 2:
                raise ValueError("EM runs on a bigram HMM; use search_ngramN for higher orders")
            return solve.iterative_em(units, self.english, {**cfg["em"], "pairs": 0}, order)  # em.pairs is for em_pairs only
        m = re.fullmatch(r"search_(ngram(\d)|neural|llm)", method)
        if m:
            sc = cfg["search"]
            candidates = None
            if m.group(1) == "llm":
                from .llm import lm_candidates

                scorer = self.llm.scorer()
                candidates = lm_candidates(self.llm, sc["top_k"])
            elif m.group(1) == "neural":
                scorer = self.llm.scorer()  # same LM, but random proposals (ablation of search_llm)
            else:
                scorer = solve.NgramScorer(self.ngram(int(m.group(2))))
            return solve.key_search(units, self.english, self._init(units, cache, sc["init"]), scorer,
                                    sc, self.seed, candidates_fn=candidates)
        if method == "llm_refine":
            from .llm import llm_refine

            rc = cfg["refine"]
            return llm_refine(units, self.english, self._init(units, cache, rc["init"]), self.llm, rc, self.seed)
        raise ValueError(f"unknown decipherment method: {method}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", help="overrides output_dir in the config")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg["seed"]
    _seed_all(seed)
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    device = _device(cfg.get("device", "auto"))
    t0 = time.time()

    # 1. data + cipher
    plain, english, vocab = data.build_splits(cfg["data"], seed)
    key = cipher.make_key(vocab, cfg["cipher"], seed)
    texts, gold = cipher.encipher(plain, key)
    _write_lines(out / "cipher.txt", texts)
    _write_lines(out / "gold_units.txt", gold)
    _write_lines(out / "plain.txt", plain)
    _write_lines(out / "english.txt", english)
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"data: {len(plain)} cipher sents, {len(english)} English sents, vocab {len(vocab)}, "
          f"{len({w for s in plain for w in s})} types in cipher text", flush=True)

    results = {"segmentation": {}, "decipherment": {}}
    seg_cfg, dec_cfg = cfg["segmentation"], cfg["decipherment"]
    alphabet_size = cfg["cipher"]["alphabet_size"]

    # 2. segmentation
    segs = {}
    for method in seg_cfg["methods"]:
        t = time.time()
        segs[method] = segment(method, texts, gold, seg_cfg, alphabet_size, device, seed)
        res = metrics.segmentation_metrics(gold, segs[method])
        res["seconds"] = round(time.time() - t, 1)
        results["segmentation"][method] = res
        _write_lines(out / f"seg_{method}.txt", segs[method])
        print(f"seg {method:<11} boundary F={res['boundary_f']:.3f} token F={res['token_f']:.3f} "
              f"lexicon F={res['lexicon_f']:.3f} ({res['seconds']}s)", flush=True)

    # 3. decipherment
    dec = Decipherer(english, dec_cfg, cfg.get("llm", {}), cfg["data"]["max_len"], device, seed, out / "llm")
    bands = [tuple(b) for b in cfg["evaluation"]["freq_bands"]]
    for seg_name in dec_cfg.get("segmentations") or list(segs):
        cache = {}
        for method in dec_cfg["methods"]:
            t = time.time()
            if method not in cache:
                cache[method] = dec.run(method, segs[seg_name], cache)
            pred_key, decoded = cache[method]
            res = metrics.decipherment_metrics(plain, decoded, key, pred_key, bands)
            res["seconds"] = round(time.time() - t, 1)
            results["decipherment"][f"{seg_name}/{method}"] = res
            _write_lines(out / f"dec_{seg_name}_{method}.txt", decoded)
            print(f"dec {seg_name:<11} {method:<14} token acc={res['token_acc']:.3f} "
                  f"vocab={res['vocab_all']:.3f} sent acc={res['sentence_acc']:.3f} "
                  f"({res['seconds']}s)", flush=True)

    results["seconds_total"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"done in {results['seconds_total']}s -> {out / 'results.json'}")
    return results


if __name__ == "__main__":
    main()
