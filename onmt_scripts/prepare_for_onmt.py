import os
import argparse

from sentencepiece import SentencePieceProcessor
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader

from summarus.readers import *


def main(train_path, val_path, test_path, config_path, subword_model_path, out_dir):
    params = Params.from_file(config_path)
    reader_params = params.pop("reader", default=Params({}))
    reader = DatasetReader.from_params(reader_params)
    processor = SentencePieceProcessor()
    processor.Load(subword_model_path)
    train_text_file = os.path.join(out_dir, "train.text.txt")
    train_summary_file = os.path.join(out_dir, "train.summary.txt")
    val_text_file = os.path.join(out_dir, "val.text.txt")
    val_summary_file = os.path.join(out_dir, "val.summary.txt")
    test_text_file = os.path.join(out_dir, "test.text.txt")
    test_summary_file = os.path.join(out_dir, "test.summary.txt")
    files = ((train_path, train_text_file, train_summary_file),
             (val_path, val_text_file, val_summary_file),
             (test_path, test_text_file, test_summary_file))
    for path, text_file_name, summary_file_name in files:
        with open(text_file_name, "w") as text_file, open(summary_file_name, "w") as summary_file:
            for text, summary in reader.parse_set(path):
                text_subwords = processor.EncodeAsPieces(text)
                summary_subwords = processor.EncodeAsPieces(summary)
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
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--subword-model-path', type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))