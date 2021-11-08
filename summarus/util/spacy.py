import json
import gc

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm


IGNORE_POS_TAGS = (
    "PUNCT",
    "CCONJ",
    "ADP",
    "PART",
    "SCONJ",
    "PRON",
    "ADV",
    "DET",
    "SYM",
    "NUM"
)
IGNORE_TOKENS = ("-", )
DOC_BIN_ATTRS = ("HEAD", "DEP", "LEMMA", "POS")


def normalize(doc, lowercase=True, ignore_pos=True, ignore_tokens=True):
    lemmas = []
    for token in doc:
        lemma = token.lemma_
        pos = token.pos_
        token = token.text
        if ignore_pos and pos in IGNORE_POS_TAGS:
            continue
        if ignore_tokens and token in IGNORE_TOKENS:
            continue
        if not token.isalpha():
            continue
        if lowercase:
            lemma = lemma.lower()
        lemmas.append(lemma)
    return lemmas


def spacy_serialize(texts, spacy_model, output_path):
    docs = DocBin(attrs=DOC_BIN_ATTRS)
    for doc in tqdm(spacy_model.pipe(texts)):
        docs.add(doc)
    gc.collect()
    docs.to_disk(output_path)


def spacy_deserialize(path):
    spacy_model = spacy.blank("ru")
    docs = DocBin().from_disk(path)
    return list(docs.get_docs(spacy_model.vocab))
