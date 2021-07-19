import json
import random
import copy
import argparse

from true_rouge import Rouge
import razdel

from summarus.util.metrics import print_metrics
from summarus.readers.gazeta_reader import parse_gazeta_json


def build_oracle_summary_greedy(text, gold_summary, calc_score, lower=True, max_sentences=30):
    output = {
        "text": text,
        "summary": gold_summary
    }
    gold_summary = gold_summary.lower() if lower else gold_summary
    original_sentences = [s.text for s in razdel.sentenize(text)]
    sentences = [s.lower() if lower else s for s in original_sentences][:max_sentences]
    def indices_to_text(indices):
        return " ".join([sentences[index] for index in sorted(list(indices))])

    n_sentences = len(sentences)

    scores = []
    final_score = -1.0
    final_indices = set()
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
    oracle_indices = [1 if i in final_indices else 0 for i in range(len(sentences))]
    output.update({
        "sentences": sentences,
        "oracle": oracle_indices
    })
    return output


def calc_single_score(pred_summary, gold_summary, rouge):
    score = rouge.get_scores([pred_summary], [gold_summary], avg=True)
    return (score['rouge-2']['f'] + score['rouge-1']['f'] + score['rouge-l']['f']) / 3


def build_oracle_records(records, nrows=None, lower=True):
    rouge = Rouge()
    calc_score = lambda x, y: calc_single_score(x, y, rouge)
    new_records = []
    for i, (text, summary) in enumerate(records):
        if nrows is not None and i >= nrows:
            break
        oracle_record = build_oracle_summary_greedy(text, summary, calc_score=calc_score, lower=lower)
        new_records.append(oracle_record)
    return new_records


def dump_records(records, file_name):
    with open(file_name, "w") as w:
        for record in records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


def main(input_path, output_path):
    records = parse_gazeta_json(input_path)
    oracle_records = build_oracle_records(records)
    dump_records(oracle_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()
    main(**vars(args))
