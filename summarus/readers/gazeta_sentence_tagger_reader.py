import json
from typing import Dict

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer

from summarus.readers.summarization_sentence_tagger_reader import SummarizationSentencesTaggerReader


def parse_gazeta_oracle_json(path):
    with open(path, "r", encoding="utf-8") as r:
        for line in r:
            data = json.loads(line.strip())
            text = data["text"]
            summary = data["summary"]
            sentences = data.get("sentences", None)
            tags = data.get("oracle", None)
            yield text, summary, sentences, tags


@DatasetReader.register("gazeta_sentences_tagger_reader")
class GazetaSentencesTaggerReader(SummarizationSentencesTaggerReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 max_sentences_count: int = 30,
                 sentence_max_tokens: int = 100,
                 lowercase: bool = True) -> None:
        super().__init__(
            tokenizer=tokenizer,
            source_token_indexers=source_token_indexers,
            max_sentences_count=max_sentences_count,
            sentence_max_tokens=sentence_max_tokens,
            lowercase=lowercase,
            language="ru"
        )

    def parse_set(self, path):
        return parse_gazeta_oracle_json(path)
