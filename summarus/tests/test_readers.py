import unittest

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import WhitespaceTokenizer

from summarus.readers import CNNDailyMailReader, RIAReader
from summarus.settings import TEST_URLS_FILE, TEST_STORIES_DIR, RIA_EXAMPLE_FILE


class TestReaders(unittest.TestCase):
    def test_cnn_dailymail_reader(self):
        tokenizer = WhitespaceTokenizer()
        reader = CNNDailyMailReader(tokenizer, cnn_tokenized_dir=TEST_STORIES_DIR, separate_namespaces=False)
        dataset = reader.read(TEST_URLS_FILE)
        for sample in dataset:
            self.assertEqual(sample.fields["source_tokens"][0].text, START_SYMBOL)
            self.assertEqual(sample.fields["source_tokens"][-1].text, END_SYMBOL)
            self.assertGreater(len(sample.fields["source_tokens"]), 2)

            self.assertEqual(sample.fields["target_tokens"][0].text, START_SYMBOL)
            self.assertEqual(sample.fields["target_tokens"][-1].text, END_SYMBOL)
            self.assertGreater(len(sample.fields["target_tokens"]), 2)

    def test_ria_reader(self):
        tokenizer = WhitespaceTokenizer()
        reader = RIAReader(tokenizer)
        dataset = reader.read(RIA_EXAMPLE_FILE)
        for sample in dataset:
            self.assertEqual(sample.fields["source_tokens"][0].text, START_SYMBOL)
            self.assertEqual(sample.fields["source_tokens"][-1].text, END_SYMBOL)
            self.assertGreater(len(sample.fields["source_tokens"]), 2)

            self.assertEqual(sample.fields["target_tokens"][0].text, START_SYMBOL)
            self.assertEqual(sample.fields["target_tokens"][-1].text, END_SYMBOL)
            self.assertGreater(len(sample.fields["target_tokens"]), 2)

    def test_ria_copy_reader(self):
        tokenizer = WhitespaceTokenizer()
        reader = RIAReader(tokenizer, separate_namespaces=True, save_copy_fields=True)
        dataset = reader.read(RIA_EXAMPLE_FILE)
        vocabulary = Vocabulary.from_instances(dataset)

        for sample in dataset:
            sample.index_fields(vocabulary)
            self.assertIsNotNone(sample.fields["source_tokens"])
            self.assertIsNotNone(sample.fields["target_tokens"])
            self.assertIsNotNone(sample.fields["metadata"].metadata)
            self.assertIsNotNone(sample.fields["source_token_ids"].array)
            self.assertIsNotNone(sample.fields["target_token_ids"].array)
            self.assertIsNotNone(sample.fields["source_to_target"]._mapping_array)
            self.assertIsNotNone(sample.fields["source_to_target"]._target_namespace)
