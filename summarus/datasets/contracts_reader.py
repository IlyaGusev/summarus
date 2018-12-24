import hashlib
import os
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


def get_article_and_abstract(story_file) -> Tuple[str, List]:
    article_lines = []
    abstract_lines = []
    next_is_highlight = False
    with open(story_file, "r", encoding="cp1251") as r:
        for line in r:
            if not line:
                continue
            elif line.startswith("@highlight"):
                next_is_highlight = True
            elif next_is_highlight:
                abstract_lines.append(line)
            else:
                article_lines.append(line)

    article = ' '.join(article_lines)
    abstract = ' '.join(abstract_lines)
    return article, [abstract]


@DatasetReader.register("contracts")
class ContractsReader(DatasetReader):
    def __init__(self,
                 contracts_dir: str,
                 tokenizer: Tokenizer = None,
                 article_token_indexers: Dict[str, TokenIndexer] = None,
                 abstract_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = True,
                 article_max_tokens: int = 400,
                 abstract_max_tokens: int = 100,
                 separate_namespaces: bool = False) -> None:
        super().__init__(lazy)

        self._contracts_dir = contracts_dir

        self._article_max_tokens = article_max_tokens
        self._abstract_max_tokens = abstract_max_tokens

        self._tokenizer = tokenizer or WordTokenizer()

        tokens_indexer = {"tokens": SingleIdTokenIndexer()}
        self._article_token_indexers = article_token_indexers or tokens_indexer
        self._abstract_token_indexers = abstract_token_indexers or tokens_indexer
        if separate_namespaces:
            self._abstract_token_indexers = abstract_token_indexers or \
                                            {"tokens": SingleIdTokenIndexer(namespace="abstract_tokens")}

    @staticmethod
    def parse_files(dir_path: str):
        file_names = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)]
        for file_name in file_names:
            yield get_article_and_abstract(file_name)

    def _read(self, dir_path: str) -> Iterable[Instance]:
        for article, abstract in self.parse_files(dir_path):
            if not article.strip() or not ''.join(abstract):
                continue
            instance = self.text_to_instance(article, abstract)
            yield instance

    def text_to_instance(self, article: str, abstract: List= None) -> Instance:
        tokenized_article = self._tokenizer.tokenize(article)[:self._article_max_tokens]
        tokenized_article.insert(0, Token(START_SYMBOL))
        tokenized_article.append(Token(END_SYMBOL))
        article_field = TextField(tokenized_article, self._article_token_indexers)

        if abstract:
            abstract = " ".join(abstract)
            tokenized_abstract = self._tokenizer.tokenize(abstract)[:self._abstract_max_tokens]
            tokenized_abstract.insert(0, Token(START_SYMBOL))
            tokenized_abstract.append(Token(END_SYMBOL))
            abstract_field = TextField(tokenized_abstract, self._abstract_token_indexers)
            return Instance({
                'source_tokens': article_field,
                'target_tokens': abstract_field
            })
        else:
            return Instance({
                'source_tokens': article_field,
            })
