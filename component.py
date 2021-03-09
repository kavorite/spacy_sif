import pdb
from collections import defaultdict, deque
from typing import Mapping

import numpy as np
import spacy
from spacy.lang.en import English
from spacy.language import Language
from spacy.tokens.doc import Doc

from embed import compute_sif_embedding


@spacy.registry.misc("enwiki_vocab_min200")
def read_weights(alpha=1e-3, default=1.0) -> Mapping[str, float]:
    freq_total = 0

    ttof = deque()
    with open("enwiki_vocab_min200.txt") as istrm:
        for line in istrm:
            term, freq = line.strip().split(maxsplit=1)
            term = term.lower()
            freq = int(freq)
            ttof.append((term, freq))
            freq_total += freq

    def ttow():
        for term, freq in ttof:
            weight = alpha / (alpha + freq / freq_total)
            yield term, weight

    return defaultdict(lambda: default, ttow())


@English.factory(
    name="sif_embed",
    assigns=["doc._.sif_tensor"],
    requires=["token.vector", "doc.sents"],
    default_config={"term_weights": {"@misc": "enwiki_vocab_min200"}},
)
class SIFEmbedder:
    def __init__(self, name: str, nlp: Language, term_weights=read_weights()):
        self.term_weights = term_weights
        self.init_component()

    def init_component(self):
        if not Doc.has_extension("sif_tensor"):
            Doc.set_extension("sif_tensor", getter=self.sif_embed)

    def sif_embed(self, doc: Doc) -> Doc:
        if not doc:
            doc._.sif_tensor = None
            return doc
        V = np.vstack([t.vector for t in doc])
        n_sents = sum(1 for sent in doc.sents)
        max_n_tokens = max(sum(1 for t in sent) for sent in doc.sents)
        x = np.zeros((n_sents, max_n_tokens), dtype="int32")
        w = np.zeros((n_sents, max_n_tokens))
        for i, sent in enumerate(doc.sents):
            for j, t in enumerate(sent):
                x[i, j] = j
                w[i, j] = self.term_weights[t.text.lower()]
        return compute_sif_embedding(V, x, w)

    def __call__(self, doc: Doc) -> Doc:
        self.init_component()
        return doc


if __name__ == "__main__":
    from itertools import combinations

    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("sif_embed")
    docs = (
        # "This is a test sentence",
        # "This is a slightly longer test sentence",
        "The Chrysler building is a skyscraper in New York. It's pretty tall.",
        "Real estate is a form of real property. It is comprised by land and any permanent improvements made on that land, including buildings.",
    )
    docs = map(nlp, docs)
    for doc1, doc2 in combinations(docs, 2):
        u = doc1._.sif_tensor[0]
        v = doc2._.sif_tensor[0]
        pdb.set_trace()
        cos = u.dot(v) / np.linalg.norm(u) / np.linalg.norm(v)
        pair = "".join(f"\n    {doc}" for doc in (doc1, doc2))
        print(f"cos({pair}) = {cos}")
