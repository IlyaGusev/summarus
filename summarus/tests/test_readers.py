import unittest

from allennlp.common.util import START_SYMBOL, END_SYMBOL

from summarus.readers import CNNDailyMailReader
from summarus.settings import TEST_URLS_FILE, TEST_STORIES_DIR


class TestReaders(unittest.TestCase):
    def test_cnn_dailymail_reader(self):
        reader = CNNDailyMailReader(cnn_tokenized_dir=TEST_STORIES_DIR, separate_namespaces=False)
        dataset = reader.read(TEST_URLS_FILE)
        for sample in dataset:
            self.assertEqual(sample.fields["source_tokens"][0].text, START_SYMBOL)
            self.assertEqual(sample.fields["source_tokens"][-1].text, END_SYMBOL)
            self.assertGreater(len(sample.fields["source_tokens"]), 2)

            self.assertEqual(sample.fields["target_tokens"][0].text, START_SYMBOL)
            self.assertEqual(sample.fields["target_tokens"][-1].text, END_SYMBOL)
            self.assertGreater(len(sample.fields["target_tokens"]), 2)
