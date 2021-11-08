import argparse
import os
from functools import lru_cache

import spacy
import numpy as np
from nltk import edit_distance
from datasets import load_dataset
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching
from sentencepiece import SentencePieceProcessor

from summarus.util.spacy import spacy_deserialize, normalize
from summarus.util.io import write_jsonl
from summarus.util.extraction_score import calc_extraction_score


class TextSummaryScorer:
    def __init__(self, vocab_file=None, spm_file=None):
        self.word2rank = dict()
        self.idfs = dict()
        self.default_idf = 0.0

        if vocab_file:
            assert os.path.exists(vocab_file)
            with open(vocab_file) as r:
                header = next(r).strip().split("\t")
                for i, line in enumerate(r):
                    row = line.strip().split("\t")
                    record = dict(zip(header, row))
                    word = record["word"].strip()
                    rank = int(record["rank"])
                    idf = float(record["idf"])
                    self.word2rank[word] = rank
                    self.idfs[word] = idf
            print("Vocabulary loaded, {} items".format(len(self.idfs)))
            self.default_idf = max(self.idfs.values())

        if spm_file:
            self.spm_tokenizer = SentencePieceProcessor()
            self.spm_tokenizer.Load(spm_file)

        self.pipeline = {
            "TokensCount": self.count_wordpiece,
            "WordRank": self.word_rank_ratio,
            "LexSim": self.lexical_similarity
        }

    def __call__(self, text, summary):
        values = dict()
        for name, action in self.pipeline.items():
            values[name] = action(text, summary)
        return values

    @staticmethod
    def levenshtein_similarity(text, summary):
        text = str(text)
        summary = str(summary)
        text = text[:len(summary)].lower()
        summary = summary.lower()
        return edit_distance(text, summary) / len(summary)

    @staticmethod
    def char_ratio(text, summary):
        text = str(text)
        summary = str(summary)
        return (len(summary) / len(text)) if len(text) != 0.0 else 0.0

    def count_wordpiece(self, text, summary):
        summary = str(summary)
        return len(self.spm_tokenizer.EncodeAsPieces(summary))

    def word_rank_ratio(self, text, summary):
        assert self.word2rank
        summary_score = self._word_rank_score(summary)
        text_score = self._word_rank_score(text)
        return summary_score / text_score if text_score != 0.0 else 0.0

    def lexical_similarity(self, text, summary):
        src_lemmas = normalize(text)
        dst_lemmas = normalize(summary)
        matching = self._get_matching(src_lemmas, dst_lemmas)
        assert len(matching) == len(src_lemmas)

        src_idf = 0.0
        for lemma, match_index in zip(src_lemmas, matching):
            if match_index != -1:
                src_idf += self.idfs.get(lemma, self.default_idf)
        dst_idf = 0.0
        for match_index in matching:
            if match_index != -1:
                lemma = dst_lemmas[match_index]
                dst_idf += self.idfs.get(lemma, self.default_idf)

        src_denominator = sum(self.idfs.get(lemma, self.default_idf) for lemma in src_lemmas)
        src_denominator += 1e-10
        dst_denominator = sum(self.idfs.get(lemma, self.default_idf) for lemma in dst_lemmas)
        dst_denominator += 1e-10

        score = 0.5 * (src_idf / src_denominator + dst_idf / dst_denominator)
        score = max(min(score, 1.), 0.)
        return score

    @staticmethod
    def summary_ttr(text, summary):
        lemmas = [token.lemma_ for token in summary]
        return len(set(lemmas)) / len(lemmas)

    @staticmethod
    def extraction_score(text, summary):
        return calc_extraction_score(str(text), str(summary))[0]

    @staticmethod
    def lcs_score(text, summary):
        return calc_extraction_score(str(text), str(summary))[1]

    def _log_rank(self, word):
        assert self.word2rank
        rank = self.word2rank.get(word, len(self.word2rank))
        return np.log(1 + rank)

    def _word_rank_score(self, text):
        assert self.word2rank
        lemmas = normalize(text)
        lemmas = [lemma for lemma in lemmas if lemma in self.word2rank]
        if len(lemmas) == 0:
            return np.log(1 + len(self.word2rank))
        ranks = [self._log_rank(lemma) for lemma in lemmas]
        return np.quantile(ranks, 0.75)

    @staticmethod
    def _get_matching(src_lemmas, dst_lemmas):
        biadjacency_matrix = np.zeros((len(src_lemmas), len(dst_lemmas)), dtype=bool)
        for i, lemma1 in enumerate(src_lemmas):
            for j, lemma2 in enumerate(dst_lemmas):
                if lemma1.lower() == lemma2.lower():
                    biadjacency_matrix[i, j] = 1
        biadjacency_matrix = sparse.csr_matrix(biadjacency_matrix)
        return maximum_bipartite_matching(biadjacency_matrix, perm_type='column')


def main(
    texts_spacy_file,
    summaries_spacy_file,
    dataset_name,
    dataset_version,
    dataset_split,
    vocab_file,
    output_path,
    spm_path,
    head
):
    dataset = load_dataset(dataset_name, script_version=dataset_version)
    dataset = list(dataset[dataset_split])

    texts_analyzes = spacy_deserialize(texts_spacy_file)
    summaries_analyzes = spacy_deserialize(summaries_spacy_file)

    assert dataset[0]["text"] == str(texts_analyzes[0])
    assert dataset[-1]["text"] == str(texts_analyzes[-1])
    assert dataset[0]["summary"] == str(summaries_analyzes[0])
    assert dataset[-1]["summary"] == str(summaries_analyzes[-1])

    scorer = TextSummaryScorer(vocab_file, spm_path)
    records = list()
    for i, (r, ta, sa) in tqdm(enumerate(zip(dataset, texts_analyzes, summaries_analyzes))):
        values = scorer(ta, sa)
        r["stats"] = values
        records.append(r)
        if head and i >= head:
            break
    write_jsonl(records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts-spacy-file", type=str, required=True)
    parser.add_argument("--summaries-spacy-file", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-version", type=str, required=True)
    parser.add_argument("--dataset-split", type=str, required=True)
    parser.add_argument("--vocab-file", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--spm-path", type=str, required=True)
    parser.add_argument("--head", type=int, default=None)
    args = parser.parse_args()
    main(**vars(args))
