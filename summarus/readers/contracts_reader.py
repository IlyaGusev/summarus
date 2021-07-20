from typing import Dict

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer

from summarus.readers.summarization_reader import SummarizationReader


@DatasetReader.register("contracts")
class ContractsReader(SummarizationReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_max_tokens: int = 400,
                 target_max_tokens: int = 100,
                 target_namespace: str = None,
                 separate_namespaces: bool = False,
                 save_copy_fields: bool = False,
                 save_pgn_fields: bool = False) -> None:
        super().__init__(
            tokenizer=tokenizer,
            source_token_indexers=source_token_indexers,
            target_token_indexers=target_token_indexers,
            source_max_tokens=source_max_tokens,
            target_max_tokens=target_max_tokens,
            target_namespace=target_namespace,
            separate_namespaces=separate_namespaces,
            save_copy_fields=save_copy_fields,
            save_pgn_fields=save_pgn_fields
        )

    def parse_set(self, file_name: str):
        with open(file_name, "r", encoding="utf-8") as r:
            text = []
            abstract = []
            is_abstract = False
            for line in r:
                line = line.strip().lower()
                if line.startswith("#new_contract#"):
                    is_abstract = False
                    if text and abstract:
                        text = " ".join(text)
                        text = text.replace("_", "")
                        abstract = " s_s ".join(abstract)
                        abstract = abstract.replace("/", "_").replace(":", "")
                        yield text, abstract
                        text = []
                        abstract = []
                elif line.startswith("#train_target_text_separator#"):
                    is_abstract = True
                elif line.startswith("text of"):
                    continue
                elif not line:
                    continue
                elif is_abstract:
                    abstract.append(line)
                else:
                    text.append(line)
