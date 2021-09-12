import os
import argparse

from sentencepiece import SentencePieceProcessor
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader

from summarus.util.io import read_jsonl


def main(
    train_path,
    val_path,
    test_path,
    subword_model_path,
    out_dir,
    max_text_subwords,
    max_summary_subwords,
    source_suffix,
    target_suffix,
    insert_tags=False,
    lowercase=False
):
    processor = SentencePieceProcessor()
    processor.Load(subword_model_path)

    train_text_file = os.path.join(out_dir, "train.{}".format(source_suffix))
    train_summary_file = os.path.join(out_dir, "train.{}".format(target_suffix))
    val_text_file = os.path.join(out_dir, "val.{}".format(source_suffix))
    val_summary_file = os.path.join(out_dir, "val.{}".format(target_suffix))
    test_text_file = os.path.join(out_dir, "test.{}".format(source_suffix))
    test_summary_file = os.path.join(out_dir, "test.{}".format(target_suffix))

    files = ((train_path, train_text_file, train_summary_file),
             (val_path, val_text_file, val_summary_file),
             (test_path, test_text_file, test_summary_file))
    for path, text_file_name, summary_file_name in files:
        with open(text_file_name, "w") as text_file, open(summary_file_name, "w") as summary_file:
            for r in read_jsonl(path):
                text = r["text"]
                summary = r["summary"]
                if lowercase:
                    text = text.lower()
                    summary = summary.lower()
                text_subwords = processor.EncodeAsPieces(text)
                if max_text_subwords:
                    text_subwords = text_subwords[:max_text_subwords]
                summary_subwords = processor.EncodeAsPieces(summary)
                if max_summary_subwords:
                    summary_subwords = summary_subwords[:max_summary_subwords]
                if insert_tags:
                    text_subwords.insert(0, "<t>")
                    text_subwords.append("</t>")
                    summary_subwords.insert(0, "<t>")
                    summary_subwords.append("</t>")
                text_file.write(" ".join(text_subwords) + "\n")
                summary_file.write((" ".join(summary_subwords)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, required=True)
    parser.add_argument('--val-path', type=str, required=True)
    parser.add_argument('--test-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--subword-model-path', type=str, required=True)
    parser.add_argument('--max-text-subwords', type=int, default=None)
    parser.add_argument('--max-summary-subwords', type=int, default=None)
    parser.add_argument('--source-suffix', type=str, default='bpe.source')
    parser.add_argument('--target-suffix', type=str, default='bpe.target')
    parser.add_argument('--insert-tags', action='store_true')
    parser.add_argument('--lowercase', action='store_true')
    args = parser.parse_args()
    main(**vars(args))
