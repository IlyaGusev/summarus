import json
from typing import Dict

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer

from summarus.readers.summarization_reader import SummarizationReader


def parse_gazeta_json(path):
    with open(path, "r", encoding="utf-8") as r:
        for line in r:
            data = json.loads(line.strip())
            summary = data["summary"]
            text = data["text"]
            if not text or not summary:
                continue
            yield text, summary


@DatasetReader.register("gazeta")
class GazetaReader(SummarizationReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_max_tokens: int = 800,
                 target_max_tokens: int = 200,
                 separate_namespaces: bool = False,
                 target_namespace: str = "target_tokens",
                 save_copy_fields: bool = False,
                 save_pgn_fields: bool = False,
                 lowercase: bool = False) -> None:
        super().__init__(
            tokenizer=tokenizer,
            source_token_indexers=source_token_indexers,
            target_token_indexers=target_token_indexers,
            source_max_tokens=source_max_tokens,
            target_max_tokens=target_max_tokens,
            separate_namespaces=separate_namespaces,
            target_namespace=target_namespace,
            save_copy_fields=save_copy_fields,
            save_pgn_fields=save_pgn_fields,
            lowercase=lowercase
        )

    def parse_set(self, path):
        return parse_gazeta_json(path)
