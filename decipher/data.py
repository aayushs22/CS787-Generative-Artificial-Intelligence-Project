"""Corpus loading, cleaning, vocabulary filtering and the cipher/English split."""

import random
import re
from collections import Counter
from pathlib import Path

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NON_WORD = re.compile(r"[^a-z' ]+")


def clean_sentence(text):
    """Lowercase, drop punctuation, return a list of word tokens."""
    text = _NON_WORD.sub(" ", text.lower().replace("’", "'"))
    return [w.strip("'") for w in text.split() if w.strip("'")]


def split_sentences(text):
    return [s for s in _SENT_SPLIT.split(text) if s.strip()]


def load_tinystories(max_stories, split="train"):
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    for i, ex in enumerate(ds):
        if i >= max_stories:
            break
        for s in split_sentences(ex["text"]):
            yield s


def load_textfile(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            for s in split_sentences(line):
                yield s


# --- Toy corpus for smoke tests (no download needed) -----------------------

_TOY = {
    "N": ["cat", "dog", "girl", "boy", "bird", "fish", "mom", "dad", "tree", "ball", "car", "cake"],
    "V": ["saw", "liked", "found", "wanted", "took", "made", "had", "lost"],
    "VI": ["ran", "played", "smiled", "jumped", "slept", "cried", "laughed"],
    "A": ["big", "little", "red", "happy", "sad", "old", "new", "small"],
    "P": ["in", "on", "with", "near", "under"],
    "D": ["the", "a", "her", "his"],
    "PN": ["lily", "tom", "ben", "sue", "max"],
}
_TOY_TEMPLATES = [
    "PN V D N", "D N VI", "D A N VI P D N", "PN V D A N", "PN and PN VI",
    "D N V D N", "PN VI P D N", "one day PN V D A N", "D N was A", "PN was very A",
    "PN said hello to D N", "they VI P D A N",
]


def toy_sentences(n, seed=0):
    rng = random.Random(seed)
    weights = {k: [1.0 / (i + 1) for i in range(len(v))] for k, v in _TOY.items()}
    for _ in range(n):
        t = rng.choice(_TOY_TEMPLATES)
        yield " ".join(
            rng.choices(_TOY[tok], weights[tok])[0] if tok in _TOY else tok for tok in t.split()
        )


def load_raw_sentences(cfg, seed):
    src = cfg["source"]
    if src == "toy":
        return toy_sentences(cfg["max_sentences"], seed)
    if src == "tinystories":
        return load_tinystories(cfg["max_stories"])
    if Path(src).exists():
        return load_textfile(src)
    raise ValueError(f"unknown data source: {src}")


def build_splits(cfg, seed):
    """Return (cipher_plain, english, vocab).

    The cleaned sentences are shuffled and split into two disjoint pools. The
    vocabulary is the top-N word types of the English pool; both pools keep
    only sentences fully inside it. `cipher_plain` is the hidden plaintext
    that gets enciphered; `english` is the non-parallel LM training text.
    """
    sents = []
    for s in load_raw_sentences(cfg, seed):
        toks = clean_sentence(s)
        if cfg["min_len"] <= len(toks) <= cfg["max_len"]:
            sents.append(toks)
    rng = random.Random(seed)
    rng.shuffle(sents)

    half = int(len(sents) * cfg["cipher_fraction"])
    cipher_pool, english_pool = sents[:half], sents[half:]

    counts = Counter(w for s in english_pool for w in s)
    vocab = {w for w, _ in counts.most_common(cfg["vocab_size"])}

    def keep(pool, n):
        out = [s for s in pool if all(w in vocab for w in s)]
        return out[:n] if n else out

    cipher_plain = keep(cipher_pool, cfg["n_cipher_sentences"])
    english = keep(english_pool, cfg["n_english_sentences"])
    if not cipher_plain or not english:
        raise ValueError("empty split: loosen vocab_size / length limits or load more data")
    return cipher_plain, english, sorted(vocab)
