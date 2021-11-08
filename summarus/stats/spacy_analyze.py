import argparse

import spacy
from datasets import load_dataset

from summarus.util.spacy import spacy_serialize


def main(
    model_name,
    dataset_name,
    dataset_version,
    dataset_split,
    output_texts_path,
    output_summaries_path
):
    spacy_model = spacy.load(model_name)
    dataset = load_dataset(dataset_name, script_version=dataset_version)

    summaries = [r["summary"] for r in dataset[dataset_split]]
    spacy_serialize(summaries, spacy_model, output_summaries_path)

    texts = [r["text"] for r in dataset[dataset_split]]
    spacy_serialize(texts, spacy_model, output_texts_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="ru_core_news_md")
    parser.add_argument("--dataset-name", type=str, default="IlyaGusev/gazeta")
    parser.add_argument("--dataset-version", type=str, required=True)
    parser.add_argument("--dataset-split", type=str, required=True)
    parser.add_argument("--output-texts-path", type=str, required=True)
    parser.add_argument("--output-summaries-path", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
