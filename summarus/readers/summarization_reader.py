from typing import Iterable, Dict, Tuple, List

import numpy as np
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField


class SummarizationReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_max_tokens: int = 400,
                 target_max_tokens: int = 100,
                 separate_namespaces: bool = False,
                 target_namespace: str = "target_tokens",
                 save_copy_fields: bool = False,
                 save_pgn_fields: bool = False,
                 lowercase: bool = True) -> None:
        super().__init__()

        assert save_pgn_fields or save_copy_fields or (not save_pgn_fields and not save_copy_fields)

        self._lowercase = lowercase
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens

        self._tokenizer = tokenizer

        tokens_indexer = {"tokens": SingleIdTokenIndexer()}
        self._source_token_indexers = source_token_indexers or tokens_indexer
        self._target_token_indexers = target_token_indexers or tokens_indexer

        self._save_copy_fields = save_copy_fields
        self._save_pgn_fields = save_pgn_fields
        self._target_namespace = "tokens"
        if separate_namespaces:
            self._target_namespace = target_namespace
            second_tokens_indexer = {"tokens": SingleIdTokenIndexer(namespace=target_namespace)}
            self._target_token_indexers = target_token_indexers or second_tokens_indexer

    def _read(self, file_path: str) -> Iterable[Instance]:
        for source, target in self.parse_set(file_path):
            if not source or not target:
                continue
            instance = self.text_to_instance(source, target)
            yield instance

    @staticmethod
    def _tokens_to_ids(tokens: List[Token], lowercase=True) -> List[int]:
        ids = dict()
        out = list()
        for token in tokens:
            token_text = token.text.lower() if lowercase else token.text
            out.append(ids.setdefault(token_text, len(ids)))
        return out

    def text_to_instance(self, source: str, target: str = None) -> Instance:
        def prepare_text(text, max_tokens):
            text = text.lower() if self._lowercase else text
            tokens = self._tokenizer.tokenize(text)[:max_tokens]
            tokens.insert(0, Token(START_SYMBOL))
            tokens.append(Token(END_SYMBOL))
            return tokens

        source_tokens = prepare_text(source, self._source_max_tokens)
        if self._save_copy_fields:
            source_tokens = source_tokens[1:-1]
        source_tokens_indexed = TextField(source_tokens, self._source_token_indexers)
        result = {'source_tokens': source_tokens_indexed}
        meta_fields = {}

        if self._save_pgn_fields or self._save_copy_fields:
            source_to_target_field = NamespaceSwappingField(source_tokens, self._target_namespace)
            result["source_to_target"] = source_to_target_field
            meta_fields["source_tokens"] = [x.text for x in source_tokens]

        if target:
            target_tokens = prepare_text(target, self._target_max_tokens)
            target_field = TextField(target_tokens, self._target_token_indexers)
            result['target_tokens'] = target_field

            if self._save_pgn_fields or self._save_copy_fields:
                meta_fields["target_tokens"] = [y.text for y in target_tokens]
                source_and_target_token_ids = self._tokens_to_ids(source_tokens + target_tokens, self._lowercase)
                source_token_ids = source_and_target_token_ids[:len(source_tokens)]
                result["source_token_ids"] = ArrayField(np.array(source_token_ids, dtype='long'))
                target_token_ids = source_and_target_token_ids[len(source_tokens):]
                result["target_token_ids"] = ArrayField(np.array(target_token_ids, dtype='long'))

        elif self._save_pgn_fields or self._save_copy_fields:
            source_token_ids = self._tokens_to_ids(source_tokens, self._lowercase)
            result["source_token_ids"] = ArrayField(np.array(source_token_ids))

        if self._save_copy_fields or self._save_pgn_fields:
            result["metadata"] = MetadataField(meta_fields)
        return Instance(result)

    def parse_set(self, path: str) -> Iterable[Tuple[str, str]]:
        raise NotImplementedError()
