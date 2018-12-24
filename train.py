import os
import logging
import argparse
from shutil import copyfile

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.training.trainer import Trainer
from allennlp.models.model import Model

from summarus.seq2seq import Seq2Seq
from summarus.datasets.cnn_dailymail_reader import CNNDailyMailReader
from summarus.settings import DEFAULT_CONFIG


def make_vocab(vocabulary_path, train_urls, val_urls, test_urls,
               cnn_dir, dm_dir, separate_namespaces=False):
    reader = CNNDailyMailReader(cnn_tokenized_dir=cnn_dir, dm_tokenized_dir=dm_dir,
                                separate_namespaces=separate_namespaces)
    train_dataset = reader.read(train_urls)
    val_dataset = reader.read(val_urls)
    test_dataset = reader.read(test_urls)
    vocabulary = Vocabulary.from_instances(test_dataset)
    vocabulary.extend_from_instances(Params({}), val_dataset)
    vocabulary.extend_from_instances(Params({}), train_dataset)
    vocabulary.save_to_files(vocabulary_path)
    return vocabulary


def train(model_path, train_urls, val_urls, test_urls, cnn_dir, dm_dir):
    vocabulary_path = os.path.join(model_path, "vocabulary")
    params_path = os.path.join(model_path, "config.json")
    params = Params.from_file(params_path)

    if os.path.exists(vocabulary_path):
        vocabulary = Vocabulary.from_files(vocabulary_path)
    else:
        vocabulary = make_vocab(vocabulary_path, train_urls, val_urls, test_urls, cnn_dir, dm_dir)

    reader = CNNDailyMailReader.from_params(params.pop("reader"), cnn_tokenized_dir=cnn_dir, dm_tokenized_dir=dm_dir)
    train_dataset = reader.read(train_urls)
    val_dataset = reader.read(val_urls)

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
         train_urls="/data/cnn_dailymail/all_train.txt",
         val_urls="/data/cnn_dailymail/all_val.txt",
         test_urls="/data/cnn_dailymail/all_test.txt",
         cnn_dir="/data/cnn_dailymail/cnn_stories_tokenized",
         dm_dir="/data/cnn_dailymail/dm_stories_tokenized"):
    assert model_name
    models_path = "models"
    model_path = os.path.join(models_path, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        copyfile(DEFAULT_CONFIG, config_path)
    train(model_path, train_urls, val_urls, test_urls, cnn_dir, dm_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--train-urls')
    parser.add_argument('--val-urls')
    parser.add_argument('--test-urls')
    parser.add_argument('--cnn-dir')
    parser.add_argument('--dm-dir')
    args = parser.parse_args()
    if not args.train_urls:
        main(args.model_name)
    else:
        main(args.model_name, args.train_urls, args.val_urls, args.test_urls, args.cnn_dir, args.dm_dir)

