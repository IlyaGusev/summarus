import unittest

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors.seq2seq import Seq2SeqPredictor

from summarus.seq2seq import Seq2Seq
from summarus.readers.cnn_dailymail_reader import CNNDailyMailReader
from summarus.settings import TEST_URLS_FILE, TEST_CONFIG, TEST_STORIES_DIR


class TestSummarizationModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = CNNDailyMailReader(cnn_tokenized_dir=TEST_STORIES_DIR, separate_namespaces=False)
        dataset = cls.reader.read(TEST_URLS_FILE)
        vocabulary = Vocabulary.from_instances(dataset)

        params = Params.from_file(TEST_CONFIG)
        model_params = params.pop("model")
        model_params.pop("type")
        cls.model = Seq2Seq.from_params(model_params, vocab=vocabulary)
        print(cls.model)
        print("Trainable params count: ", sum(p.numel() for p in cls.model.parameters() if p.requires_grad))

        iterator = DataIterator.from_params(params.pop('iterator'))
        iterator.index_with(vocabulary)
        trainer = Trainer.from_params(cls.model, None, iterator,
                                      dataset, None, params.pop('trainer'))
        trainer.train()

    def test_model(self):
        self.model.training = False
        predictor = Seq2SeqPredictor(self.model, self.reader)
        for article, reference_sents in self.reader.parse_set(TEST_URLS_FILE):
            decoded_words = predictor.predict(article)["predicted_tokens"]
            self.assertListEqual(decoded_words, reference_sents.split())

