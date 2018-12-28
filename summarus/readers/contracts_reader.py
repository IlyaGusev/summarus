import os
from typing import Dict

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer

from summarus.readers.summarization_reader import SummarizationReader
from summarus.readers.cnn_dailymail_reader import get_article_and_abstract


@DatasetReader.register("contracts")
class ContractsReader(SummarizationReader):
    def __init__(self,
                 contracts_dir: str,
                 tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_max_tokens: int = 400,
                 target_max_tokens: int = 100,
                 separate_namespaces: bool = False) -> None:
        super().__init__(
            tokenizer=tokenizer,
            source_token_indexers=source_token_indexers,
            target_token_indexers=target_token_indexers,
            source_max_tokens=source_max_tokens,
            target_max_tokens=target_max_tokens,
            separate_namespaces=separate_namespaces
        )

        self._contracts_dir = contracts_dir

    def parse_set(self, dir_path: str):
        file_names = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)]
        for file_name in file_names:
            yield get_article_and_abstract(file_name, encoding="cp1251")
