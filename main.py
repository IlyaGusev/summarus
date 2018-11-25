import os

import rouge
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.common.params import Params
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.training.trainer import Trainer
from allennlp.models.model import Model
from allennlp.predictors.simple_seq2seq import SimpleSeq2SeqPredictor

from summarus.datasets.cnn_dailymail_reader import CNNDailyMailReader

train_urls = "/data/cnn_dailymail/all_train.txt"
val_urls = "/data/cnn_dailymail/all_val.txt"
test_urls = "/data/cnn_dailymail/all_test.txt"
cnn_dir = "/data/cnn_dailymail/cnn_stories_tokenized"
dm_dir = "/data/cnn_dailymail/dm_stories_tokenized"
train_cache = "/data/cnn_dailymail/train.pickle"
val_cache = "/data/cnn_dailymail/val.pickle"
reader = CNNDailyMailReader(cnn_dir, dm_dir, lazy=True)
train_dataset = reader.read(train_urls)
val_dataset = reader.read(val_urls)
test_dataset = reader.read(test_urls)

models_path = "models"
model_path = os.path.join(models_path, "25_11_2018_2")
vocabulary_path = os.path.join(model_path, "vocabulary")
params_path = os.path.join(model_path, "config.json")

# train_short_dataset = []
# for instance in train_dataset:
#     train_short_dataset.append(instance)
#     if len(train_short_dataset) == 1000:
#         break
# vocabulary = Vocabulary.from_instances(train_dataset, max_vocab_size=50000)
# vocabulary.save_to_files("vocabulary")

vocabulary = Vocabulary.from_files(vocabulary_path)
params = Params.from_file(params_path)
model = SimpleSeq2Seq.from_params(params.pop("model"), vocab=vocabulary, target_namespace="abstract_tokens")
print(model)
print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

iterator = DataIterator.from_params(params.pop('iterator'))
iterator.index_with(vocabulary)
trainer = Trainer.from_params(model, model_path, iterator,
                              train_dataset, val_dataset, params.pop('trainer'))
trainer.train()


# vocabulary = Vocabulary.from_files(vocabulary_path)
# params = Params.from_file(params_path)
# model = Model.load(params, model_path, cuda_device=0)
#
# all_hypothesis = []
# all_references = []
# predictor = SimpleSeq2SeqPredictor(model, reader)
# eval_path = os.path.join(model_path, "eval.txt")
# with open(eval_path, "w", encoding="utf-8") as w:
#     for article, abstract in reader.parse_files(test_urls):
#         pred_abstract = " ".join(predictor.predict(article)["predicted_tokens"])
#         all_hypothesis.append(pred_abstract)
#         all_references.append(abstract)
#         w.write(abstract + "\t" + pred_abstract + "\n")
#
# evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
#                         max_n=4,
#                         limit_length=True,
#                         length_limit=100,
#                         length_limit_type='words',
#                         stemming=False)
# scores = evaluator.get_scores(all_hypothesis, all_references)
# print(scores)


