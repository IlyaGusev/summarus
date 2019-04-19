import unittest
import os
import torch
import numpy as np

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.common.params import Params
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors.seq2seq import Seq2SeqPredictor
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.models.model import Model

from summarus.settings import TEST_URLS_FILE, TEST_CONFIG_DIR, TEST_STORIES_DIR, RIA_EXAMPLE_FILE


class TestSummarizationModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed = 1337
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.set_flags(True, False, True, True)
        cls.params = []
        for file_name in os.listdir(TEST_CONFIG_DIR):
            if not file_name.endswith(".json"):
                continue
            config_path = os.path.join(TEST_CONFIG_DIR, file_name)
            cls.params.append(Params.from_file(config_path))

    def test_models(self):
        for params in self.params:
            reader_params = params.duplicate().pop("reader", default=Params({}))
            if reader_params["type"] == "cnn_dailymail":
                reader_params["cnn_tokenized_dir"] = TEST_STORIES_DIR
                dataset_file = TEST_URLS_FILE
            elif reader_params["type"] == "ria":
                dataset_file = RIA_EXAMPLE_FILE
            else:
                assert False

            reader = DatasetReader.from_params(reader_params)
            tokenizer = reader._tokenizer
            dataset = reader.read(dataset_file)
            vocabulary_params = params.pop("vocabulary", default=Params({}))
            vocabulary = Vocabulary.from_params(vocabulary_params, instances=dataset)

            model_params = params.pop("model")
            model = Model.from_params(model_params, vocab=vocabulary)
            print(model)
            print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

            iterator = DataIterator.from_params(params.pop('iterator'))
            iterator.index_with(vocabulary)
            trainer = Trainer.from_params(model, None, iterator,
                                          dataset, None, params.pop('trainer'))
            trainer.train()

            model.training = False
            predictor = Seq2SeqPredictor(model, reader)
            for article, reference_sents in reader.parse_set(dataset_file):
                ref_words = [token.text for token in tokenizer.tokenize(reference_sents)]
                decoded_words = predictor.predict(article)["predicted_tokens"]
                self.assertGreaterEqual(len(decoded_words), len(ref_words))
                while DEFAULT_OOV_TOKEN in decoded_words:
                    unk_index = decoded_words.index(DEFAULT_OOV_TOKEN)
                    decoded_words.pop(unk_index)
                    if unk_index < len(ref_words):
                        ref_words.pop(unk_index)
                self.assertListEqual(decoded_words[:len(ref_words)], ref_words)

