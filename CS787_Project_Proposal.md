# CS787 Project Proposal: Generative Priors for Unsupervised Word Discovery and Decipherment

## 1. Introduction and Motivation

Machine translation typically relies on parallel corpora, bilingual dictionaries, or other forms of linguistic supervision. For low-resource, historical, or undeciphered languages, such resources may be unavailable. This motivates methods that can discover linguistic structure and recover meaning from text using only a *generative model of the target language* as a guide.

Two research directions address parts of this problem. Goldwater, Griffiths, and Johnson (2006) propose Bayesian models (Dirichlet process unigram and hierarchical Dirichlet process bigram) for discovering word boundaries in continuous text, and show that modelling word-to-word dependencies is crucial for avoiding under-segmentation of frequent collocations. [Read the paper](https://aclanthology.org/P06-1085/). Ravi and Knight (2011) treat foreign text as a cipher for English and recover the hidden word-substitution key using EM and Bayesian inference guided by an English n-gram language model. [Read the paper](https://aclanthology.org/P11-1002/).

These problems are usually studied separately: decipherment assumes pre-segmented input, while segmentation stops at boundaries. Both, however, are driven by the same ingredient — a probabilistic model of what plausible language looks like. In the original papers that model is a Dirichlet process or an n-gram LM. In this project we ask whether **modern neural generative models can serve as stronger priors for both stages**, and we study the full pipeline from unsegmented symbols to recovered English.

## 2. Problem Statement

We construct a controlled synthetic cipher from an English corpus. Each distinct English word type is assigned a consistent, randomly generated code, and word boundaries are removed. For example:

```text
English:   the cat eats fish
           the cat likes fish
Key:       the → kq, cat → bmr, eats → t, likes → bmqa, fish → rrk
Cipher:    kqbmrtrrk
           kqbmrbmqarrk
```

Codes have variable length and are drawn from a small shared alphabet, so the same symbol sequence can be read in several ways (e.g. `bmr` versus `bmqa` share a prefix), and segmentation is genuinely ambiguous.

The system receives only (i) the unsegmented cipher corpus, with sentence boundaries given, and (ii) a separate, non-parallel English corpus from the same domain. It does not receive the word boundaries, the key, a bilingual dictionary, or parallel sentences. The goal is to recover the word boundaries and then decipher the text into English. Ground-truth boundaries and the key are retained for evaluation only.

Because the key is a one-to-one relabelling of English, this setting is a **controlled upper bound** on real decipherment: it isolates the difficulty of segmentation and of key recovery from the additional difficulties of real languages (non-deterministic translation, reordering, morphology).

## 3. Data Design

- **Corpus.** We use TinyStories (Eldan and Li, 2023), a large corpus of simple English stories with a small core vocabulary. The cipher is generated from one split; all English models are trained on a disjoint split, so plaintext and cipher are never parallel.
- **Vocabulary control.** We keep only sentences whose words all fall within the top *N* word types, at two difficulty levels: *N* ≈ 500 (comparable to the 153-type temporal-expression corpus of Ravi and Knight) and *N* ≈ 2,000 (comparable to their 3,397-type Transtac corpus).
- **Preprocessing.** Text is lowercased and punctuation is removed. Sentence boundaries are given to the system, following the setting of Goldwater et al. (2006).
- **Encoding.** Each word type receives a code of random length (1–5 symbols) over an alphabet of about 20 symbols. Codes are unique but not required to be prefix-free. Code length is independent of the English word; a length-correlated encoding is studied as an ablation.

## 4. Proposed Methodology

### 4.1 Stage 1: Unsupervised Word-Boundary Discovery

We compare a classical Bayesian approach with a neural generative approach.

- **Bayesian segmentation (Goldwater et al., 2006).** We implement the Dirichlet process unigram model and the hierarchical Dirichlet process bigram model, with Gibbs sampling over boundary positions and simulated annealing, following Section 3 and Section 4 of the paper.
- **Neural character language model.** We train a small character-level transformer (or LSTM) on the cipher text alone and place boundaries where next-symbol surprisal or predictive entropy rises sharply. This is a neural realisation of the classical "successor variety" principle that transitions within words are more predictable than transitions across them.

### 4.2 Stage 2: Decipherment

Given a segmentation, we recover the key using three methods with progressively stronger generative priors.

1. **Frequency baseline.** Match cipher types to English types by frequency rank.
2. **Iterative EM with an n-gram LM (Ravi and Knight, 2011, Section 2.1).** The cipher is modelled as an HMM whose transitions are a fixed English bigram or trigram LM and whose emissions are the unknown channel P(c|e). To remain tractable, the channel is grown iteratively over the most frequent types, fixing confident mappings at each stage, and the final decipherment is obtained by Viterbi decoding.
3. **Neural-LM key search.** We train a small GPT-style language model on the English split. Since the key is one-to-one, we search directly over keys: starting from the EM solution, simulated annealing proposes reassignments and swaps of cipher-to-English mappings, and each candidate key is scored by the neural LM's log-likelihood of the resulting decoded corpus (or a sample of it). This lets a neural model, which cannot be used inside forward–backward, act as the decipherment prior.

**Stretch goals.**

- **Embedding alignment.** Train word embeddings independently on the segmented cipher and on English, and align the two spaces without supervision via adversarial initialisation and Procrustes refinement (Conneau et al., 2018). Because the cipher is an exact relabelling of English, the spaces are isomorphic up to sampling noise, which makes this a clean test of representation-based decipherment.
- **LM infilling cleanup.** Low-confidence positions are re-predicted by a masked or infilling LM, with a vote across all occurrences of each cipher type to keep the key consistent.

### 4.3 Handling Segmentation Errors

Automatic segmentation can merge frequent collocations into a single unit or split a word into several units, which breaks the one-to-one assumption. We therefore allow the decipherment key to map a cipher unit to an English word, an English word pair, or `UNK`. This makes the effect of segmentation errors measurable rather than silently propagating them.

### 4.4 Integrated Pipeline

1. **Unsegmented cipher text:** variable-length symbol codes with no word boundaries.
2. **Segmentation:** Bayesian sampler or neural character LM.
3. **Decipherment:** frequency baseline, Iterative EM, or neural-LM key search.
4. **Recovered English text:** evaluated against the hidden ground truth.

## 5. Research Questions

- **RQ1: Can neural generative models discover words?** Does a character-level neural LM, trained only on the cipher, segment as well as the Bayesian DP/HDP models of Goldwater et al. (2006)?
- **RQ2: How much do segmentation errors cost?** How much decipherment accuracy is lost when using predicted boundaries instead of oracle boundaries, and how does this loss grow with deliberately perturbed boundaries?
- **RQ3: Does a neural prior help decipherment?** How do unigram, bigram, and trigram LMs compare with a neural LM as the decipherment prior, and is the neural advantage larger when cipher data is scarce?
- **RQ4: How much data is required?** How do segmentation and decipherment change as the cipher corpus grows from 1,000 to 5,000 to 20,000 sentences?

## 6. Experimental Design

**Experiment 1: Segmentation.** Compare the DP unigram sampler, the HDP bigram sampler, and the neural character LM. We expect the unigram model to under-segment frequent collocations, as reported by Goldwater et al., and test whether context-aware models (HDP and neural) avoid this.

**Experiment 2: Oracle versus predicted segmentation.** Run each decipherment method under three conditions: oracle boundaries, boundaries from the best segmenter, and oracle boundaries with controlled perturbation (randomly merging or splitting a fixed fraction of words).

**Experiment 3: Language-model prior.** Compare the frequency baseline, Iterative EM with unigram, bigram, and trigram LMs, and neural-LM key search, at both vocabulary sizes.

**Experiment 4: Data scaling and ablations.** Vary the cipher corpus size (1k, 5k, 20k sentences). As an ablation, use a length-correlated encoding to measure how much information code length leaks to the segmenter.

## 7. Evaluation Metrics

- **Segmentation:** boundary precision, recall, and F1; word-token F1 and lexicon F1 as defined by Goldwater et al. (2006), for direct comparison with the paper.
- **Decipherment token accuracy:** the proportion of English tokens recovered correctly, computed after an edit-distance alignment between the decoded and reference sentences so that it remains well defined when the predicted segmentation has a different number of units.
- **Vocabulary recovery:** the proportion of word types whose mapping is recovered correctly, reported separately by frequency band (frequent, medium, rare), since singleton words are nearly undecipherable by any LM-driven method.
- **Sentence accuracy:** the proportion of sentences deciphered exactly.

Ground-truth boundaries and keys are used only for evaluation.

## 8. Expected Contributions

1. A controlled benchmark for joint segmentation and decipherment, with tunable vocabulary size, corpus size, and encoding.
2. A comparison of classical Bayesian and neural generative models for unsupervised word discovery.
3. A method for using a neural language model as the prior in word-substitution decipherment, compared against the n-gram EM approach of Ravi and Knight (2011).
4. A quantitative account of how segmentation errors, prior strength, and data size each limit end-to-end recovery of English.

## 9. Scope and Timeline

| Week | Planned work | Deliverable |
|---|---|---|
| 1 | Cipher generator, frequency baseline, DP unigram sampler | Dataset and first segmentation results |
| 2 | HDP bigram sampler, neural character-LM segmenter, Iterative EM with oracle boundaries | Segmentation comparison and working EM decipherment |
| 3 | Train English GPT on TinyStories, neural-LM key search, full pipeline integration | End-to-end system |
| 4 | Oracle/predicted/perturbed experiments, LM comparison, data scaling; stretch goals if time permits | Main experimental results |
| 5 | Error analysis, ablations, report, and presentation | Final report and presentation |

The essential deliverables are the cipher benchmark, the Bayesian and neural segmenters, Iterative EM and neural-LM decipherment, and the experiments for RQ1–RQ4. Embedding alignment, LM infilling, many-to-one keys, and joint segmentation–decipherment (scoring a segmentation lattice with the English LM) are stretch goals.

## 10. References

### Base Papers

- Goldwater, S., Griffiths, T. L., & Johnson, M. (2006). **Contextual Dependencies in Unsupervised Word Segmentation.** In *Proceedings of COLING/ACL 2006*, pp. 673–680. [ACL Anthology](https://aclanthology.org/P06-1085/)
- Ravi, S., & Knight, K. (2011). **Deciphering Foreign Language.** In *Proceedings of ACL-HLT 2011*, pp. 12–21. [ACL Anthology](https://aclanthology.org/P11-1002/)

### Supporting References

- Conneau, A., Lample, G., Ranzato, M., Denoyer, L., & Jégou, H. (2018). **Word Translation Without Parallel Data.** In *ICLR 2018*. [arXiv:1710.04087](https://arxiv.org/abs/1710.04087)
- Eldan, R., & Li, Y. (2023). **TinyStories: How Small Can Language Models Be and Still Speak Coherent English?** [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)

## 11. Conclusion

This project investigates whether an unknown text can be deciphered when even its word boundaries are unavailable, and whether modern generative models make this easier. Building on Bayesian segmentation and statistical decipherment, we replace or complement their hand-built priors with neural language models and measure the effect at each stage.

Our central hypothesis is that **stronger generative priors improve both word discovery and decipherment, but segmentation errors remain the main bottleneck when no parallel data or dictionary is available**. The main outcome will be an empirical analysis of how segmentation quality, prior strength, and data size jointly determine the success of the full pipeline.
