from typing import Iterable, Dict, Tuple

from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField


class SummarizationReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_max_tokens: int = 400,
                 target_max_tokens: int = 100,
                 separate_namespaces: bool = False) -> None:
        super().__init__(lazy=True)

        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens

        self._tokenizer = tokenizer or WordTokenizer()

        tokens_indexer = {"tokens": SingleIdTokenIndexer()}
        self._source_token_indexers = source_token_indexers or tokens_indexer
        self._target_token_indexers = target_token_indexers or tokens_indexer

        if separate_namespaces:
            second_tokens_indexer = {"tokens": SingleIdTokenIndexer(namespace="target_tokens")}
            self._target_token_indexers = target_token_indexers or second_tokens_indexer

    def _read(self, file_path: str) -> Iterable[Instance]:
        for source, target in self.parse_set(file_path):
            assert source and target
            instance = self.text_to_instance(source, target)
            yield instance

    def text_to_instance(self, source: str, target: str = None) -> Instance:
        def prepare_text(text, max_tokens):
            tokens = self._tokenizer.tokenize(text)[:max_tokens]
            tokens.insert(0, Token(START_SYMBOL))
            tokens.append(Token(END_SYMBOL))
            return tokens

        source_tokens = prepare_text(source, self._source_max_tokens)
        source_tokens_indexed = TextField(source_tokens, self._source_token_indexers)
        result = {'source_tokens': source_tokens_indexed}

        if target:
            target_tokens = prepare_text(target, self._target_max_tokens)
            target_tokens_indexed = TextField(target_tokens, self._target_token_indexers)
            result['target_tokens'] = target_tokens_indexed

        return Instance(result)

    def parse_set(self, path: str) -> Iterable[Tuple[str, str]]:
        raise NotImplementedError()
