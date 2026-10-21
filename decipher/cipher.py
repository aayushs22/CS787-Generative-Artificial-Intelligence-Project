"""Synthetic word-substitution cipher: key generation and encoding."""

import math
import random
import string

SYMBOLS = string.ascii_uppercase + string.digits


def make_key(vocab, cfg, seed):
    """Assign each English word type a unique random code.

    length_mode="random": code length uniform in [min_code_len, max_code_len],
    independent of the word (default). length_mode="correlated": length grows
    with the English word length (ablation that leaks length information).
    """
    k = cfg["alphabet_size"]
    if not 2 <= k <= len(SYMBOLS):
        raise ValueError(f"alphabet_size must be in [2, {len(SYMBOLS)}]")
    alphabet = SYMBOLS[:k]
    lo, hi = cfg["min_code_len"], cfg["max_code_len"]
    capacity = sum(k ** n for n in range(lo, hi + 1))
    if capacity < len(vocab):
        raise ValueError(f"only {capacity} codes available for {len(vocab)} words")

    rng = random.Random(seed)
    words = list(vocab)
    rng.shuffle(words)
    key, used = {}, set()
    for w in words:
        if cfg["length_mode"] == "correlated":
            n = min(hi, max(lo, math.ceil(len(w) / 2)))
        else:
            n = rng.randint(lo, hi)
        for _ in range(1000):
            code = "".join(rng.choice(alphabet) for _ in range(n))
            if code not in used:
                break
            n = min(hi, n + 1)  # length bucket full: move to a longer one
        else:
            raise RuntimeError("could not find an unused code")
        used.add(code)
        key[w] = code
    return key


def encipher(sentences, key):
    """Return (texts, gold_units): unsegmented strings and their gold segmentation."""
    gold = [[key[w] for w in s] for s in sentences]
    return ["".join(units) for units in gold], gold
