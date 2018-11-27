import os

from rouge import Rouge
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.training.trainer import Trainer
from allennlp.models.model import Model
from allennlp.predictors.simple_seq2seq import SimpleSeq2SeqPredictor

from summarus.seq2seq import Seq2Seq
from summarus.datasets.cnn_dailymail_reader import CNNDailyMailReader

train_urls = "/data/cnn_dailymail/all_train.txt"
val_urls = "/data/cnn_dailymail/all_val.txt"
test_urls = "/data/cnn_dailymail/all_test.txt"
cnn_dir = "/data/cnn_dailymail/cnn_stories_tokenized"
dm_dir = "/data/cnn_dailymail/dm_stories_tokenized"
train_cache = "/data/cnn_dailymail/train.pickle"
val_cache = "/data/cnn_dailymail/val.pickle"


def train(model_name):
    models_path = "models"
    model_path = os.path.join(models_path, model_name)
    vocabulary_path = os.path.join(model_path, "vocabulary")
    params_path = os.path.join(model_path, "config.json")

    reader = CNNDailyMailReader(cnn_dir, dm_dir, lazy=True)
    train_dataset = reader.read(train_urls)
    val_dataset = reader.read(val_urls)

    if os.path.exists(vocabulary_path):
        vocabulary = Vocabulary.from_files(vocabulary_path)
    else:
        vocabulary = Vocabulary.from_instances(train_dataset)
        vocabulary.save_to_files(vocabulary_path)
        train_dataset = reader.read(train_urls)

    params = Params.from_file(params_path)
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


def evaluate(model_name):
    models_path = "models"
    model_path = os.path.join(models_path, model_name)
    params_path = os.path.join(model_path, "config.json")
    eval_path = os.path.join(model_path, "eval.txt")

    reader = CNNDailyMailReader(cnn_dir, dm_dir, lazy=True)

    params = Params.from_file(params_path)
    model = Model.load(params, model_path, cuda_device=0)
    print(model)
    print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    all_hypothesis = []
    all_references = []

    if not os.path.exists(eval_path):
        predictor = SimpleSeq2SeqPredictor(model, reader)
        with open(eval_path, "w", encoding="utf-8") as w:
            for article, abstract in reader.parse_files(test_urls):
                abstract = " ".join(abstract.split(" ")[:100])
                print("Article: ", article)
                print("Abstract: ", abstract)
                pred_abstract = " ".join(predictor.predict(article)["predicted_tokens"])
                print("Pred abstract: ", pred_abstract)
                all_hypothesis.append(pred_abstract)
                all_references.append(abstract)
                w.write(abstract + "\t" + pred_abstract + "\n")
    else:
        with open(eval_path, "r", encoding="utf-8") as r:
            for line in r:
                abstract, pred_abstract = line.split("\t")
                all_hypothesis.append(pred_abstract)
                all_references.append(abstract)

    evaluator = Rouge()
    scores = evaluator.get_scores(all_hypothesis, all_references, avg=True)
    print(scores)


evaluate("single_layer_lstm_adam")
