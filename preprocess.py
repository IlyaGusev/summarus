import os
import argparse

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from sentencepiece import SentencePieceTrainer as sp_trainer

from summarus.readers import *


def preprocess(train_path, vocabulary_path, config_path):
    params = Params.from_file(config_path)

    reader_params = params.pop("reader", default=Params({}))
    vocabulary_params = params.pop("vocabulary", default=Params({}))

    if "tokenizer" in reader_params:
        tokenizer = reader_params["tokenizer"]
        if tokenizer["type"] == "subword":
            assert "max_vocab_size" in vocabulary_params
            max_vocab_size = int(vocabulary_params["max_vocab_size"])
            if not os.path.exists(vocabulary_path):
                os.makedirs(vocabulary_path)
            cmd = "--input={} --model_prefix={} --vocab_size={} --model_type={}".format(
                train_path,
                os.path.join(vocabulary_path, "bpe"),
                max_vocab_size,
                "bpe")
            sp_trainer.Train(cmd)

    reader = DatasetReader.from_params(reader_params)
    dataset = reader.read(train_path)

    vocabulary = Vocabulary.from_params(vocabulary_params, instances=dataset)
    vocabulary.save_to_files(vocabulary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-path', required=True)
    parser.add_argument('--vocabulary-path', required=True)
    parser.add_argument('--config-path', required=True)
    args = parser.parse_args()
    preprocess(**vars(args))
