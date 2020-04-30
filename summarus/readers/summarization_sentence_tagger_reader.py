from typing import Iterable, Dict, Tuple, List

from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import Tokenizer, Token, WhitespaceTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.fields import TextField, ListField, SequenceLabelField


class SummarizationSentencesTaggerReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 max_sentences_count: int = 100,
                 sentence_max_tokens: int = 100,
                 lowercase: bool = True) -> None:
        super().__init__(lazy=True)

        self._lowercase = lowercase
        self._max_sentences_count = max_sentences_count
        self._sentence_max_tokens = sentence_max_tokens
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:
        for _, _, sentences, tags in self.parse_set(file_path):
            if sum(tags) == 0:
                continue
            assert len(sentences) == len(tags)
            instance = self.text_to_instance(sentences, tags)
            yield instance

    def text_to_instance(self, sentences: List[str], tags: List[int] = None) -> Instance:
        sentences_tokens = []
        for sentence in sentences[:self._max_sentences_count]:
            sentence = sentence.lower() if self._lowercase else sentence
            tokens = self._tokenizer.tokenize(sentence)[:self._sentence_max_tokens]
            tokens.insert(0, Token(START_SYMBOL))
            tokens.append(Token(END_SYMBOL))
            indexed_tokens = TextField(tokens, self._source_token_indexers)
            sentences_tokens.append(indexed_tokens)

        sentences_tokens_indexed = ListField(sentences_tokens)
        result = {'source_sentences': sentences_tokens_indexed}

        if tags:
            result["sentences_tags"] = SequenceLabelField(tags[:self._max_sentences_count], sentences_tokens_indexed)
        return Instance(result)

    def parse_set(self, path: str) -> Iterable[Tuple[List[str], List[int]]]:
        raise NotImplementedError()
