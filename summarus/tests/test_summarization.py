import random
import unittest
import os
import tempfile
import shutil
import torch
import numpy as np

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.common.params import Params
from allennlp.data.data_loaders import DataLoader
from allennlp.training.trainer import Trainer
from allennlp_models.generation.predictors.seq2seq import Seq2SeqPredictor
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.models.model import Model

from summarus.settings import TEST_URLS_FILE, TEST_CONFIG_DIR, TEST_STORIES_DIR, RIA_EXAMPLE_FILE


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TestSummarizationModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(1337)
        cls.params = {}
        for file_name in os.listdir(TEST_CONFIG_DIR):
            if not file_name.endswith(".json"):
                continue
            config_path = os.path.join(TEST_CONFIG_DIR, file_name)
            cls.params[file_name] = Params.from_file(config_path)

    def _test_model(self, file_name):
        params = self.params[file_name].duplicate()
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
        loader = DataLoader.from_params(
            reader=reader,
            data_path=dataset_file,
            params=params.pop("data_loader")
        )

        instances = reader.read(dataset_file)
        vocabulary_params = params.pop("vocabulary", default=Params({}))
        vocabulary = Vocabulary.from_params(vocabulary_params, instances=instances)

        loader.index_with(vocabulary)

        model_params = params.pop("model")
        model = Model.from_params(model_params, vocab=vocabulary)
        print(model)
        print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        temp_dir = tempfile.mkdtemp()
        trainer = Trainer.from_params(
            params.pop('trainer'),
            model=model,
            data_loader=loader,
            serialization_dir=temp_dir
        )
        trainer.train()

        model.eval()
        predictor = Seq2SeqPredictor(model, reader)
        for article, reference_sents in reader.parse_set(dataset_file):
            ref_words = [token.text for token in tokenizer.tokenize(reference_sents)]
            decoded_words = predictor.predict(article)["predicted_tokens"][0]
            print("REF: ", ref_words)
            print("HYP: ", decoded_words)
            self.assertGreaterEqual(len(decoded_words), len(ref_words))
            unk_count = 0
            while DEFAULT_OOV_TOKEN in decoded_words:
                unk_index = decoded_words.index(DEFAULT_OOV_TOKEN)
                decoded_words.pop(unk_index)
                unk_count += 1
                if unk_index < len(ref_words):
                    ref_words.pop(unk_index)
            self.assertLess(unk_count, 6)
            self.assertListEqual(decoded_words[:len(ref_words)], ref_words)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_ria_pgn(self):
        self._test_model("ria_pgn.json")

    def test_cnn_dm_seq2seq(self):
        self._test_model("cnn_dm_seq2seq.json")

    def test_cnn_dm_pgn(self):
        self._test_model("cnn_dm_pgn.json")

    def test_cnn_dm_copynet(self):
        self._test_model("cnn_dm_copynet.json")
