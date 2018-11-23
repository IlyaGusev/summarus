from allennlp.data.vocabulary import Vocabulary
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.common.params import Params
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.training.trainer import Trainer

from summarus.datasets.cnn_dailymail_reader import CNNDailyMailReader

train_urls = "/media/yallen/My Passport/Datasets/Summarization/cnn_dailymail/all_train.txt"
val_urls = "/media/yallen/My Passport/Datasets/Summarization/cnn_dailymail/all_val.txt"
test_urls = "/media/yallen/My Passport/Datasets/Summarization/cnn_dailymail/all_test.txt"
cnn_dir = "/media/yallen/My Passport/Datasets/Summarization/cnn_dailymail/cnn_stories_tokenized"
dm_dir = "/media/yallen/My Passport/Datasets/Summarization/cnn_dailymail/dm_stories_tokenized"
reader = CNNDailyMailReader(cnn_dir, dm_dir, lazy=True)
train_dataset = reader.read(train_urls)
val_dataset = reader.read(train_urls)

train_short_dataset = []
for instance in train_dataset:
    train_short_dataset.append(instance)
    if len(train_short_dataset) > 10000:
        break
vocabulary = Vocabulary.from_instances(train_short_dataset, max_vocab_size=5000)
print(vocabulary.get_vocab_size())

params = Params.from_file("config.json")
model = SimpleSeq2Seq.from_params(params.pop("model"), vocab=vocabulary)
print(model)
print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

train_dataset = reader.read(train_urls)
iterator = DataIterator.from_params(params.pop('iterator'))
iterator.index_with(vocabulary)
trainer = Trainer.from_params(model, "model", iterator,
                              train_dataset, val_dataset, params.pop('trainer'))
trainer.train()