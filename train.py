import os
import argparse
from shutil import copyfile

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.training.trainer import Trainer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from summarus.seq2seq import Seq2Seq
from summarus.readers.cnn_dailymail_reader import CNNDailyMailReader
from summarus.readers.contracts_reader import ContractsReader
from summarus.settings import DEFAULT_CONFIG


def train(model_path, train_path, val_path):
    params_path = os.path.join(model_path, "config.json")
    params = Params.from_file(params_path)

    reader_params = params.duplicate().pop("reader", default=Params({}))
    reader = DatasetReader.from_params(reader_params)
    train_dataset = reader.read(train_path)

    vocabulary_path = os.path.join(model_path, "vocabulary")
    vocabulary_params = params.pop("vocabulary", default=Params({}))
    if os.path.exists(vocabulary_path):
        vocabulary = Vocabulary.from_files(vocabulary_path)
    else:
        vocabulary = Vocabulary.from_params(vocabulary_params, instances=train_dataset)
        vocabulary.save_to_files(vocabulary_path)

    val_dataset = reader.read(val_path) if val_path else None

    model_params = params.pop("model")
    model_params.pop("type")
    model = Seq2Seq.from_params(model_params, vocab=vocabulary)
    print(model)
    print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    iterator = DataIterator.from_params(params.pop('iterator'))
    iterator.index_with(vocabulary)
    trainer = Trainer.from_params(model, model_path, iterator,
                                  train_dataset, val_dataset, params.pop('trainer'))
    trainer.train()


def main(model_name,
         train_path="/data/cnn_dailymail/all_train.txt",
         val_path="/data/cnn_dailymail/all_val.txt"):
    assert model_name
    models_path = "models"
    model_path = os.path.join(models_path, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        copyfile(DEFAULT_CONFIG, config_path)
    train(model_path, train_path, val_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--train-path')
    parser.add_argument('--val-path')
    args = parser.parse_args()
    if not args.train_path:
        main(args.model_name)
    else:
        main(args.model_name, args.train_path, args.val_path)

