import json
import copy
import argparse

from true_rouge import Rouge
import razdel
from nltk.translate.chrf_score import sentence_chrf
from tqdm import tqdm

from summarus.readers.gazeta_reader import parse_gazeta_json


def build_oracle_summary_greedy(text, gold_summary, calc_score, max_sentences=40):
    original_sentences = [s.text for s in razdel.sentenize(text)]
    sentences = original_sentences[:max_sentences]

    def indices_to_text(indices):
        return " ".join([sentences[index] for index in sorted(list(indices))])

    scores = []
    final_score = -1.0
    final_indices = set()
    n_sentences = len(sentences)
    for _ in range(n_sentences):
        for i in range(n_sentences):
            if i in final_indices:
                continue
            indices = copy.copy(final_indices)
            indices.add(i)
            summary = indices_to_text(indices)
            scores.append((calc_score(summary, gold_summary), indices))

        # If metrics didn't increase in outer loop, stop
        best_score, best_indices = max(scores)
        scores = []
        if best_score <= final_score:
            break
        final_score, final_indices = best_score, best_indices

    oracle_indices = [int(i in final_indices) for i in range(len(sentences))]
    return {
        "text": text,
        "summary": gold_summary,
        "sentences": sentences,
        "oracle": oracle_indices
    }


def build_oracle_records(records, metric, lower, nrows=None):
    rouge = Rouge()

    def calc_score(x, y):
        if lower:
            x = x.lower()
            y = y.lower()
        if metric == "rouge":
            score = rouge.get_scores([x], [y], avg=True)
            return (score['rouge-2']['f'] + score['rouge-1']['f'] + score['rouge-l']['f']) / 3
        elif metric == "chrf":
            return sentence_chrf(x, y, beta=1.5)

    new_records = []
    for i, (text, summary) in enumerate(tqdm(records)):
        if nrows is not None and i >= nrows:
            break
        oracle_record = build_oracle_summary_greedy(text, summary, calc_score=calc_score)
        new_records.append(oracle_record)
    return new_records


def dump_records(records, file_name):
    with open(file_name, "w") as w:
        for record in records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


def main(input_path, output_path, metric, lower):
    records = parse_gazeta_json(input_path)
    oracle_records = build_oracle_records(records, metric, lower)
    dump_records(oracle_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--metric', required=True, choices=("chrf", "rouge"))
    parser.add_argument('--lower', action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
