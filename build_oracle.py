import json
import random
import copy
import argparse

from rouge import Rouge
import razdel

from summarus.util.metrics import print_metrics


def read_gazeta_records(file_name, shuffle=True, sort_by_date=False):
    assert shuffle != sort_by_date
    records = []
    with open(file_name, "r") as r:
        for line in r:
            records.append(json.loads(line))
    if sort_by_date:
        records.sort(key=lambda x: x["date"])
    if shuffle:
        random.shuffle(records)
    return records


def build_oracle_summary_greedy(text, gold_summary, calc_score, lower=True, max_sentences=30):
    gold_summary = gold_summary.lower() if lower else gold_summary
    # Делим текст на предложения
    original_sentences = [sentence.text for sentence in razdel.sentenize(text)]
    sentences = [sentence.text.lower() if lower else sentence.text for sentence in razdel.sentenize(text)][:max_sentences]
    n_sentences = len(sentences)
    oracle_summary_sentences = set()
    score = -1.0
    summaries = []
    for _ in range(n_sentences):
        for i in range(n_sentences):
            if i in oracle_summary_sentences:
                continue
            current_summary_sentences = copy.copy(oracle_summary_sentences)
            # Добавляем какое-то предложения к уже существующему summary
            current_summary_sentences.add(i)
            current_summary = " ".join([sentences[index] for index in sorted(list(current_summary_sentences))])
            # Считаем метрики
            current_score = calc_score(current_summary, gold_summary)
            summaries.append((current_score, current_summary_sentences))
        # Если получилось улучшить метрики с добавлением какого-либо предложения, то пробуем добавить ещё
        # Иначе на этом заканчиваем
        best_summary_score, best_summary_sentences = max(summaries)
        if best_summary_score <= score:
            break
        oracle_summary_sentences = best_summary_sentences
        score = best_summary_score
    oracle_summary = " ".join([sentences[index] for index in sorted(list(oracle_summary_sentences))])
    return oracle_summary, original_sentences, oracle_summary_sentences


def calc_single_score(pred_summary, gold_summary, rouge):
    score = rouge.get_scores([pred_summary], [gold_summary], avg=True)
    return (score['rouge-2']['f'] + score['rouge-1']['f'] + score['rouge-l']['f']) / 3


def build_oracle_records(records, nrows=None, lower=True):
    references = []
    predictions = []
    rouge = Rouge()
    new_records = []

    for i, record in enumerate(records):
        if nrows is not None and i >= nrows:
            break

        summary = record["summary"]
        summary = summary if not lower else summary.lower()
        references.append(summary)

        text = record["text"]
        predicted_summary, sentences, oracle_indices = build_oracle_summary_greedy(text, summary, calc_score=lambda x,
                                                                                                                    y: calc_single_score(
            x, y, rouge))
        predictions.append(predicted_summary)
        oracle_indices = [1 if i in oracle_indices else 0 for i in range(len(sentences))]

        new_record = copy.copy(record)
        new_record["sentences"] = sentences
        new_record["oracle"] = oracle_indices
        new_records.append(new_record)

    print_metrics(references, predictions)
    return new_records


def dump_records(records, file_name):
    with open(file_name, "w") as w:
        for record in records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


def main(input_path, output_path):
    records = read_gazeta_records(input_path)
    dump_records(build_oracle_records(records), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()
    main(**vars(args))
