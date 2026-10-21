"""Neural segmenter: boundaries where a character LM trained on the cipher is uncertain."""

import math

from .. import neural
from .baselines import scores_to_segments


def charlm_segment(texts, alphabet, cfg, device, seed):
    vocab = neural.Vocab(alphabet)
    seqs = [vocab.encode(list(s)) for s in texts]
    model = neural.build_model(len(vocab), cfg["model"])
    neural.train_lm(model, seqs, cfg["train"], device, seed, tag="charlm")
    ents = neural.predictive_entropy(model, seqs, device, cfg["train"]["batch_size"])
    # ents[k][t] = H(symbol t | symbols < t); boundary before symbol i scored by ents[k][i].
    scores = [[0.0 if math.isnan(e) else e for e in ent[:len(s)]] for s, ent in zip(texts, ents)]
    return scores_to_segments(texts, scores, cfg["quantile"])
