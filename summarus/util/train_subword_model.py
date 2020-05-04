import os
import tempfile
import argparse

from sentencepiece import SentencePieceTrainer as sp_trainer
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader

from summarus.readers import *


def train_subwords(train_path, model_path, model_type,
                   vocab_size, config_path, lowercase=False):
    temp = tempfile.NamedTemporaryFile(mode="w", delete=False)
    params = Params.from_file(config_path)
    reader_params = params.pop("dataset_reader", default=Params({}))
    reader = DatasetReader.from_params(reader_params)
    for text, summary in reader.parse_set(train_path):
        text = text.lower() if lowercase else text
        summary = summary.lower() if lowercase else summary
        temp.write(text + "\n")
        temp.write(summary + "\n")
    temp.close()
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    cmd = "--input={} --model_prefix={} --vocab_size={} --model_type={}".format(
        temp.name,
        os.path.join(model_path, model_type),
        vocab_size,
        model_type)
    sp_trainer.Train(cmd)
    os.unlink(temp.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--model-type', type=str, default="bpe")
    parser.add_argument('--vocab-size', type=int, default=50000)
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--lowercase', action='store_true')
    args = parser.parse_args()
    train_subwords(**vars(args))
