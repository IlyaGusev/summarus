import os
import argparse
from shutil import copyfile

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.training.trainer import Trainer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.models.model import Model

from summarus.seq2seq import Seq2Seq
from summarus.readers.cnn_dailymail_reader import CNNDailyMailReader
from summarus.readers.contracts_reader import ContractsReader
from summarus.settings import DEFAULT_CONFIG


def train(model_path, train_path, val_path, vocabulary_path=None, config_path=None):
    params_path = config_path or os.path.join(model_path, "config.json")
    params = Params.from_file(params_path)

    vocabulary_path = vocabulary_path or os.path.join(model_path, "vocabulary")
    assert os.path.exists(vocabulary_path), "Vocabulary is not ready, run preprocess.py first"
    vocabulary = Vocabulary.from_files(vocabulary_path)

    reader_params = params.duplicate().pop("reader", default=Params({}))
    reader = DatasetReader.from_params(reader_params)
    train_dataset = reader.read(train_path)
    val_dataset = reader.read(val_path) if val_path else None

    model_params = params.pop("model")
    model = Model.from_params(model_params, vocab=vocabulary)
    print(model)
    print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    iterator = DataIterator.from_params(params.pop('iterator'))
    iterator.index_with(vocabulary)
    trainer = Trainer.from_params(model, model_path, iterator,
                                  train_dataset, val_dataset, params.pop('trainer'))
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--train-path', required=True)
    parser.add_argument('--val-path', default=None)
    parser.add_argument('--vocabulary-path', default=None)
    parser.add_argument('--config-path', default=None)
    args = parser.parse_args()
    train(**vars(args))

