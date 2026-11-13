# Results

TinyStories, sentences of 3–12 words, 20-symbol alphabet, code lengths 1–5. English LM data is disjoint from the cipher pool
(20,000 sentences for N=500, 60,000 for N=2000). Configs are in `configs/`, raw output in `runs/` (git-ignored); `python -m decipher.report runs/<name>` prints a table.

| run | vocab N | cipher sentences | seed | notes |
|---|---|---|---|---|
| `n500_c1000` | 500 | 1,000 | 0 | all segmenters and all 7 decipherment methods |
| `n500_c1000_s1`, `_s2` | 500 | 1,000 | 1, 2 | 5 methods, 5,000 search steps |
| `n500_c5000` | 500 | 5,000 | 0 | 5 methods |
| `n500_c20000` | 500 | 20,000 | 0 | 5 methods |
| `n2000_c5000` | 2,000 | 5,000 | 0 | 5 methods |
| `n500_c1000_corr` | 500 | 1,000 | 0 | length-correlated codes (ablation) |

**Caveats.** Only the 1k setting has three seeds; every other cell is a single run. Learned-segmenter results vary a lot between seeds
(at 1k with EM: `hdp_bigram` 0.13 / 0.40 / 0.24 and `charlm` 0.17 / 0.11 / 0.09 for seeds 0 / 1 / 2), so segmenter rankings and anything
involving them are suggestive only. Samplers, the char-LM and the other models were not tuned (Gibbs: 100 iterations at every corpus size, which is probably
too few for 20k sentences). The `perturbed`/`entropy` segmenters and `search_ngram3`/`search_llm` were run only in `n500_c1000`.

## Findings by research question

- **RQ2 – cost of segmentation errors (clearest result).** With oracle boundaries EM reaches 0.89 (1k, 3-seed mean), 0.97 (5k), 0.985 (20k) and 0.84 (N=2000, 5k).
  With 10% random merges/splits it drops to 0.72 (seed 0). With *learned* segmentation it is 0.09–0.55 at 1k, 0.20–0.34 at 5k, 0.08–0.42 at 20k and 0.23–0.42 at N=2000. Accuracy with learned boundaries does not follow corpus size the way oracle accuracy does, so segmentation is the bottleneck, as hypothesised.
- **RQ1 – neural vs Bayesian segmentation.** At 1k sentences the char-LM is the worst segmenter in all three seeds (boundary F 0.56–0.60 vs 0.76–0.86 for DP/HDP). It improves with data (0.72 at 5k, 0.71 at 20k) while DP/HDP *get worse* (DP 0.81 → 0.78 → 0.70) — likely under-convergence of the fixed-iteration samplers, which we did not check. At 20k the char-LM gives the best downstream accuracy (EM 0.42 vs 0.08 DP / 0.17 HDP), although its lexicon F is low (0.12), i.e. it finds many spurious unit types. DP unigram did not clearly under-segment relative to HDP bigram; it was often better at 1k.
- **RQ3 – neural prior.** With oracle boundaries at 1k (3 seeds), mean token accuracy: EM 0.887, n-gram key search 0.891, neural-LM key search 0.894, `llm_refine` 0.903. `llm_refine` beats EM in each of the three seeds, by 1.2–2.0 points; the key-search variants are within noise of each other. The same ordering holds at 5k (0.968 EM vs 0.980 `llm_refine`) and 20k (0.985 vs 0.994), and N=2000 (0.836 vs 0.850). Gains are small because EM is already near the ceiling here. Under learned segmentation, no method is consistently ahead. We did not test whether the neural advantage is larger with scarce cipher data below 1k sentences. Note `llm_refine` keeps its starting key unless the LM score improves, so it can only match or beat its EM init on LM score, not necessarily on accuracy.
- **RQ4 – data size.** Oracle-boundary accuracy rises with cipher size: EM 0.89 → 0.97 → 0.985 for 1k → 5k → 20k. Segmentation does not improve correspondingly (see RQ1).
- **Vocabulary size.** N=2000 with 5k sentences: oracle EM 0.84 vs 0.97 for N=500 at the same size, so the bigger vocabulary is harder, as expected. EM at this size takes about 12 minutes per segmentation.
- **Length-correlated codes.** Oracle results are unchanged (0.884 EM). Segmentation F changed by only a few points, but downstream accuracy with learned segmentations moved in both directions by amounts as large as seed noise, so there is no clear evidence for or against length leaking into the segmenter.
- **Key search vs its EM start.** n-gram key search did not reliably improve on EM (oracle 1k: 0.877 / 0.899 / 0.898 vs 0.884 / 0.889 / 0.887).

## Bug fixed while running these

At 20k sentences the HDP bigram sampler crashed (`TypeError: '<' not supported between 'complex' and 'float'`). Accumulated float error left an expected table count slightly negative (about −2e-16); for rare long units the prior term is ~1e-27, so the probability went negative and annealing (`p ** (1/temp)`) made it complex. `decipher/segment/gibbs.py` now clamps that count at 0. The 1k and 5k runs completed without hitting this; I did not rerun them after the fix.

## Word-pair keys (§4.3) and embedding alignment

Runs `ext_c1000_s{0,1,2}`, `ext_c5000`, `ext_c20000` (same data and seeds as the `n500_*` runs; `ext_c20000` skips `hdp_bigram`). Token accuracy:

| setting | seg | em | em_pairs (50 pair states) | embed_align |
|---|---|---|---|---|
| 1k, seeds 0/1/2 | oracle | .884 / .889 / .887 | .717 / .762 / .770 | .059 / .067 / .044 |
| 1k, seeds 0/1/2 | dp_unigram | .394 / .553 / .397 | .550 / .690 / .601 | .018 / .031 / .023 |
| 5k | oracle | .968 | .888 | .109 |
| 5k | dp_unigram | .200 | .256 | .024 |
| 20k | oracle | .985 | .904 | .600 |
| 20k | dp_unigram | .083 | .070 | .025 |

- **Word-pair keys help when units are fused and hurt otherwise.** With DP unigram segmentation at 1k, `em_pairs` is better than `em` in all three seeds, by 14–20 points. With oracle boundaries it costs 11–17 points, because the extra pair states add spurious alternatives. At 5k the gain is smaller (+5.6) and at 20k it is gone (−1.3). With HDP and char-LM boundaries it is mixed (e.g. HDP at 1k: +4.5, −19.5, −8.7 points for seeds 0/1/2). The number of pair states (50) was not tuned and the method has no way to decide per unit whether a pair is warranted, so it cannot be a default.
- **Embedding alignment is weak at this scale.** Below the frequency baseline (0.23–0.29) at 1k and 5k even with oracle boundaries, 0.60 at 20k (above the frequency baseline of 0.39, far below EM's 0.985), and near zero with learned segmentation. It needs far more cipher data than EM. Its settings (dim 30, 300 types, 100 init types) were picked by trying four combinations on the 5k and 20k oracle data, i.e. on the evaluation data, so the 20k number is optimistic; results are single-run.

## Tables

### runs/n500_c1000

**Segmentation**

| method | boundary F | token F | lexicon F |
|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 |
| perturbed | 0.947 | 0.864 | 0.717 |
| entropy | 0.731 | 0.541 | 0.375 |
| dp_unigram | 0.811 | 0.646 | 0.642 |
| hdp_bigram | 0.785 | 0.556 | 0.508 |
| charlm | 0.598 | 0.367 | 0.259 |

**Decipherment token accuracy (rows: segmentation)**

| seg | freq | em | search_ngram2 | search_ngram3 | search_neural | search_llm | llm_refine |
|---|---|---|---|---|---|---|---|
| oracle | 0.229 | 0.884 | 0.877 | 0.851 | 0.887 | 0.891 | 0.896 |
| perturbed | 0.116 | 0.718 | 0.705 | 0.724 | 0.719 | 0.699 | 0.732 |
| entropy | 0.061 | 0.398 | 0.414 | 0.443 | 0.399 | 0.402 | 0.410 |
| dp_unigram | 0.055 | 0.394 | 0.404 | 0.407 | 0.372 | 0.372 | 0.405 |
| hdp_bigram | 0.050 | 0.125 | 0.147 | 0.148 | 0.116 | 0.184 | 0.123 |
| charlm | 0.060 | 0.166 | 0.190 | 0.168 | 0.181 | 0.136 | 0.172 |

### runs/n500_c1000_s1

**Segmentation**

| method | boundary F | token F | lexicon F |
|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 |
| dp_unigram | 0.857 | 0.716 | 0.703 |
| hdp_bigram | 0.791 | 0.573 | 0.557 |
| charlm | 0.556 | 0.272 | 0.251 |

**Decipherment token accuracy (rows: segmentation)**

| seg | freq | em | search_ngram2 | search_neural | llm_refine |
|---|---|---|---|---|---|
| oracle | 0.240 | 0.889 | 0.899 | 0.891 | 0.907 |
| dp_unigram | 0.069 | 0.553 | 0.568 | 0.555 | 0.587 |
| hdp_bigram | 0.035 | 0.403 | 0.445 | 0.482 | 0.443 |
| charlm | 0.068 | 0.111 | 0.147 | 0.121 | 0.110 |

### runs/n500_c1000_s2

**Segmentation**

| method | boundary F | token F | lexicon F |
|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 |
| dp_unigram | 0.821 | 0.655 | 0.646 |
| hdp_bigram | 0.762 | 0.499 | 0.514 |
| charlm | 0.555 | 0.293 | 0.253 |

**Decipherment token accuracy (rows: segmentation)**

| seg | freq | em | search_ngram2 | search_neural | llm_refine |
|---|---|---|---|---|---|
| oracle | 0.246 | 0.887 | 0.898 | 0.903 | 0.907 |
| dp_unigram | 0.080 | 0.397 | 0.413 | 0.436 | 0.415 |
| hdp_bigram | 0.066 | 0.235 | 0.314 | 0.257 | 0.256 |
| charlm | 0.025 | 0.088 | 0.100 | 0.090 | 0.085 |

### runs/n500_c5000

**Segmentation**

| method | boundary F | token F | lexicon F |
|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 |
| dp_unigram | 0.782 | 0.561 | 0.665 |
| hdp_bigram | 0.764 | 0.561 | 0.545 |
| charlm | 0.720 | 0.527 | 0.217 |

**Decipherment token accuracy (rows: segmentation)**

| seg | freq | em | search_ngram2 | search_neural | llm_refine |
|---|---|---|---|---|---|
| oracle | 0.292 | 0.968 | 0.966 | 0.970 | 0.980 |
| dp_unigram | 0.037 | 0.200 | 0.239 | 0.215 | 0.224 |
| hdp_bigram | 0.048 | 0.325 | 0.337 | 0.342 | 0.330 |
| charlm | 0.056 | 0.341 | 0.354 | 0.351 | 0.366 |

### runs/n500_c20000

**Segmentation**

| method | boundary F | token F | lexicon F |
|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 |
| dp_unigram | 0.700 | 0.402 | 0.436 |
| hdp_bigram | 0.692 | 0.470 | 0.425 |
| charlm | 0.713 | 0.541 | 0.120 |

**Decipherment token accuracy (rows: segmentation)**

| seg | freq | em | search_ngram2 | search_neural | llm_refine |
|---|---|---|---|---|---|
| oracle | 0.393 | 0.985 | 0.957 | 0.972 | 0.994 |
| dp_unigram | 0.034 | 0.083 | 0.084 | 0.082 | 0.102 |
| hdp_bigram | 0.024 | 0.166 | 0.190 | 0.176 | 0.174 |
| charlm | 0.035 | 0.417 | 0.399 | 0.404 | 0.412 |

### runs/n2000_c5000

**Segmentation**

| method | boundary F | token F | lexicon F |
|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 |
| dp_unigram | 0.836 | 0.657 | 0.611 |
| hdp_bigram | 0.763 | 0.512 | 0.450 |
| charlm | 0.670 | 0.454 | 0.280 |

**Decipherment token accuracy (rows: segmentation)**

| seg | freq | em | search_ngram2 | search_neural | llm_refine |
|---|---|---|---|---|---|
| oracle | 0.310 | 0.836 | 0.835 | 0.828 | 0.850 |
| dp_unigram | 0.122 | 0.423 | 0.430 | 0.436 | 0.439 |
| hdp_bigram | 0.047 | 0.320 | 0.362 | 0.351 | 0.354 |
| charlm | 0.022 | 0.227 | 0.292 | 0.264 | 0.229 |

### runs/n500_c1000_corr

**Segmentation**

| method | boundary F | token F | lexicon F |
|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 |
| dp_unigram | 0.765 | 0.531 | 0.591 |
| hdp_bigram | 0.761 | 0.537 | 0.525 |
| charlm | 0.565 | 0.331 | 0.271 |

**Decipherment token accuracy (rows: segmentation)**

| seg | freq | em | search_ngram2 | search_neural | llm_refine |
|---|---|---|---|---|---|
| oracle | 0.229 | 0.884 | 0.885 | 0.887 | 0.896 |
| dp_unigram | 0.105 | 0.156 | 0.150 | 0.158 | 0.174 |
| hdp_bigram | 0.022 | 0.384 | 0.379 | 0.359 | 0.397 |
| charlm | 0.056 | 0.143 | 0.146 | 0.152 | 0.146 |

### runs/ext_c1000_s0

**Segmentation**

| method | boundary F | token F | lexicon F |
|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 |
| dp_unigram | 0.811 | 0.646 | 0.642 |
| hdp_bigram | 0.785 | 0.556 | 0.508 |
| charlm | 0.598 | 0.367 | 0.259 |

**Decipherment token accuracy (rows: segmentation)**

| seg | em | em_pairs | embed_align |
|---|---|---|---|
| oracle | 0.884 | 0.717 | 0.059 |
| dp_unigram | 0.394 | 0.550 | 0.018 |
| hdp_bigram | 0.125 | 0.170 | 0.035 |
| charlm | 0.166 | 0.127 | 0.005 |

### runs/ext_c5000

**Segmentation**

| method | boundary F | token F | lexicon F |
|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 |
| dp_unigram | 0.782 | 0.561 | 0.665 |
| hdp_bigram | 0.764 | 0.561 | 0.545 |
| charlm | 0.720 | 0.527 | 0.217 |

**Decipherment token accuracy (rows: segmentation)**

| seg | em | em_pairs | embed_align |
|---|---|---|---|
| oracle | 0.968 | 0.888 | 0.109 |
| dp_unigram | 0.200 | 0.256 | 0.024 |
| hdp_bigram | 0.325 | 0.112 | 0.011 |
| charlm | 0.341 | 0.195 | 0.018 |

### runs/ext_c20000

**Segmentation**

| method | boundary F | token F | lexicon F |
|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 |
| dp_unigram | 0.700 | 0.402 | 0.436 |
| charlm | 0.713 | 0.541 | 0.120 |

**Decipherment token accuracy (rows: segmentation)**

| seg | em | em_pairs | embed_align |
|---|---|---|---|
| oracle | 0.985 | 0.904 | 0.600 |
| dp_unigram | 0.083 | 0.070 | 0.025 |
| charlm | 0.417 | 0.394 | 0.011 |
