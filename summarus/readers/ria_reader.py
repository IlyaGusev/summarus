import json
from typing import Dict

from bs4 import BeautifulSoup
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer

from summarus.readers.summarization_reader import SummarizationReader


def parse_ria_json(path):
    with open(path, "r", encoding="utf-8") as r:
        for line in r:
            data = json.loads(line.strip())
            title = data["title"]
            text = data["text"]
            clean_text = BeautifulSoup(text, 'html.parser').text.replace('\xa0', ' ').replace('\n', ' ')
            if not clean_text or not title:
                continue
            yield clean_text, title


@DatasetReader.register("ria")
class RIAReader(SummarizationReader):
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
        return parse_ria_json(path)
