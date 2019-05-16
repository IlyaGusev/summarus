import os
import argparse

import numpy
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.training.trainer import Trainer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.models.model import Model

from summarus import *


def set_seed(seed):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model_path, train_path, val_path, seed, vocabulary_path=None, config_path=None, pretrained_path=None):
    assert os.path.isdir(model_path), "Model directory does not exist"
    set_seed(seed)

    config_path = config_path or os.path.join(model_path, "config.json")
    assert os.path.isfile(config_path), "Config file does not exist"
    params = Params.from_file(config_path)

    vocabulary_path = vocabulary_path or os.path.join(model_path, "vocabulary")
    assert os.path.exists(vocabulary_path), "Vocabulary is not ready, do not forget to run preprocess.py first"
    vocabulary = Vocabulary.from_files(vocabulary_path)

    reader_params = params.duplicate().pop("reader", default=Params({}))
    reader = DatasetReader.from_params(reader_params)
    train_dataset = reader.read(train_path)
    val_dataset = reader.read(val_path) if val_path else None

    if not pretrained_path:
        model_params = params.pop("model")
        model = Model.from_params(model_params, vocab=vocabulary)
    else:
        model = Model.load(params, pretrained_path)
    print(model)
    print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    iterator = DataIterator.from_params(params.pop('iterator'))
    iterator.index_with(vocabulary)
    trainer = Trainer.from_params(model, model_path, iterator,
                                  train_dataset, val_dataset, params.pop('trainer'))
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for model training")
    parser.add_argument('--model-path', required=True, help="path to directory with model's files")
    parser.add_argument('--train-path', required=True)
    parser.add_argument('--val-path', default=None)
    parser.add_argument('--seed', type=int, default=1048596)
    parser.add_argument('--vocabulary-path', default=None)
    parser.add_argument('--config-path', default=None)
    parser.add_argument('--pretrained-path', default=None)
    args = parser.parse_args()
    train(**vars(args))

