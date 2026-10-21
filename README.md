# CS787: Generative Priors for Unsupervised Word Discovery and Decipherment

English text is enciphered by replacing each word type with a random variable-length symbol code and deleting word boundaries. The pipeline then (1) recovers word boundaries without supervision and (2) recovers the hidden word-to-code key using only a separate, non-parallel English corpus. See [CS787_Project_Proposal.md](CS787_Project_Proposal.md) for the motivation and research questions.

## Setup

```bash
pip install -r requirements.txt   # torch, numpy, pyyaml, pytest; datasets for TinyStories
```

## Running

```bash
python -m decipher.run --config configs/smoke.yaml            # toy corpus, ~5 s on CPU
python -m decipher.run --config configs/smoke.yaml --output-dir runs/try1
pytest -q tests                                                # unit + end-to-end tests
```

`configs/smoke.yaml` uses a built-in toy grammar and tiny models so the full pipeline runs in seconds. **Its values are placeholders, not tuned settings.** For real runs, copy it, set `data.source: tinystories` (or a path to a text file), and choose your own sizes and hyperparameters.

Each run writes to `output_dir`:

| File | Contents |
|---|---|
| `cipher.txt`, `plain.txt`, `gold_units.txt`, `key.json` | the cipher text and its hidden ground truth (evaluation only) |
| `english.txt` | the non-parallel English text used for every LM |
| `seg_<method>.txt` | predicted segmentation, units separated by spaces |
| `dec_<seg>_<method>.txt` | decoded English for each segmentation × decipherment pair |
| `results.json` | all metrics and timings |
| `config.yaml` | copy of the config used |

## Pipeline

| Stage | Methods (config names) | Code |
|---|---|---|
| Data | `toy`, `tinystories`, text file; disjoint cipher/English pools; top-N vocabulary filter | [decipher/data.py](decipher/data.py) |
| Cipher | random-length codes (`length_mode: random`) or length-correlated ablation (`correlated`) | [decipher/cipher.py](decipher/cipher.py) |
| Segmentation | `oracle`, `perturbed` (controlled merge/split rate), `entropy` (branching entropy), `dp_unigram` and `hdp_bigram` (Goldwater et al. 2006 Gibbs samplers), `charlm` (neural char-LM entropy) | [decipher/segment/](decipher/segment/) |
| Decipherment | `freq` (rank matching), `em` / `em1` / `em2` (Iterative EM, Ravi & Knight 2011 §2.1), `search_ngramN` and `search_neural` (annealed key search scored by an N-gram or neural English LM) | [decipher/solve.py](decipher/solve.py) |
| LMs | Witten–Bell word n-gram; small causal transformer | [decipher/ngram.py](decipher/ngram.py), [decipher/neural.py](decipher/neural.py) |
| Metrics | boundary / token / lexicon P-R-F; aligned token accuracy, normalised edit distance, sentence accuracy, vocabulary recovery per frequency band | [decipher/metrics.py](decipher/metrics.py) |

Every segmentation listed in `segmentation.methods` is passed to every method in `decipherment.methods` (restrict this with `decipherment.segmentations`). Key search starts from the `em` or `freq` key (`search.init`).

## Implementation notes

- **Gibbs samplers** use the corrected equations from Goldwater, Griffiths & Johnson (2009), because the 2006 appendix has typos in the bigram denominators. The bigram sampler replaces table counts with their expected values, as in the 2006 appendix. Both are written in pure Python, so expect minutes to hours per run at 10k+ sentences; porting the inner loop to numba is the obvious speed-up.
- **Iterative EM** runs a bigram HMM, so `em` supports LM orders 1–2. Higher-order and neural LMs are used through key search. Forward–backward is batched over sentences of equal length, and Viterbi is restricted to a beam of candidate English words per cipher unit.
- **The neural architecture** in `neural.py` is a minimal placeholder. Any model mapping `(B, T)` ids to `(B, T, vocab)` logits can replace it.

## Not yet implemented (from the proposal)

- §4.3 mapping a cipher unit to an English *word pair*. Currently, merged units can only map to a single word, and the alignment-based metrics count the resulting errors.
- Stretch goals: embedding alignment, LM-infilling cleanup, many-to-one keys, joint segmentation–decipherment.
