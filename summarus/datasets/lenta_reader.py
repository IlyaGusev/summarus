import os
import csv
from typing import Iterable, Dict, List, Tuple

from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField


@DatasetReader.register("lenta")
class LentaReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 text_token_indexers: Dict[str, TokenIndexer] = None,
                 title_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = True,
                 text_max_tokens: int = 600,
                 title_max_tokens: int = 40,
                 separate_namespaces: bool = False) -> None:
        super().__init__(lazy)

        self._text_max_tokens = text_max_tokens
        self._title_max_tokens = title_max_tokens

        self._tokenizer = tokenizer or WordTokenizer()

        tokens_indexer = {"tokens": SingleIdTokenIndexer()}
        self._text_token_indexers = text_token_indexers or tokens_indexer
        self._title_token_indexers = title_token_indexers or tokens_indexer
        if separate_namespaces:
            self._title_token_indexers = title_token_indexers or \
                                         {"tokens": SingleIdTokenIndexer(namespace="title_tokens")}

    def _read(self, file_path: str) -> Iterable[Instance]:
        for text, title in self.parse_files(file_path):
            instance = self.text_to_instance(text, title)
            yield instance

    def text_to_instance(self, text: str, title: str = None) -> Instance:
        tokenized_text = self._tokenizer.tokenize(text)[:self._text_max_tokens]
        tokenized_text.insert(0, Token(START_SYMBOL))
        tokenized_text.append(Token(END_SYMBOL))
        text_field = TextField(tokenized_text, self._text_token_indexers)

        if title:
            tokenized_title = self._tokenizer.tokenize(title)[:self._title_max_tokens]
            tokenized_title.insert(0, Token(START_SYMBOL))
            tokenized_title.append(Token(END_SYMBOL))
            title_field = TextField(tokenized_title, self._title_token_indexers)
            return Instance({
                'source_tokens': text_field,
                'target_tokens': title_field
            })
        else:
            return Instance({
                'source_tokens': text_field,
            })

    def parse_files(self, path):
         with open(file_path, "r", encoding="utf-8") as r:
            reader = csv.reader(r, delimiter=",", quotechar='"')
            header = next(reader)
            assert header[1] == "title"
            assert header[2] == "text"
            for row in reader:
                title, text = row[1], row[2]
                if not text or not title:
                    continue
                yield text, title
