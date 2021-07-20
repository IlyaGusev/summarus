import hashlib
import os
from typing import Iterable, Dict, Tuple

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer

from summarus.readers.summarization_reader import SummarizationReader

dm_single_close_quote = u'\u2019'
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"]


def fix_missing_period(line):
    if "@highlight" in line:
        return line
    elif not line:
        return line
    elif line[-1] in END_TOKENS:
        return line
    return line + " ."


def hashhex(s):
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def get_file_names_by_urls(cnn_tokenized_dir, dm_tokenized_dir, urls_file_path):
    with open(urls_file_path, "r", encoding="utf-8") as r:
        for url in r:
            url = url.strip()
            file_name = str(hashhex(url)) + ".story"
            dirs = (cnn_tokenized_dir, dm_tokenized_dir)
            file_names = [os.path.join(d, file_name) for d in dirs if d is not None]
            file_found = False
            for f in file_names:
                if os.path.isfile(f):
                    file_found = True
                    yield f
                    break
            assert file_found, "File not found in tokenized dir: " + file_name


def get_article_and_abstract(story_file, encoding="utf-8", fix_period=True) -> Tuple[str, str]:
    article_lines = []
    abstract = []
    next_is_highlight = False
    with open(story_file, "r", encoding=encoding) as r:
        for line in r:
            line = line.strip().lower()
            if fix_period:
                line = fix_missing_period(line)
            if not line:
                continue
            elif line.startswith("@highlight"):
                next_is_highlight = True
            elif next_is_highlight:
                abstract.append(line)
            else:
                article_lines.append(line)

    article = ' '.join(article_lines)
    abstract = ' s_s '.join(abstract)
    return article, abstract


@DatasetReader.register("cnn_dailymail")
class CNNDailyMailReader(SummarizationReader):
    def __init__(self,
                 tokenizer: Tokenizer,
                 cnn_tokenized_dir: str = None,
                 dm_tokenized_dir: str = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_max_tokens: int = 400,
                 target_max_tokens: int = 100,
                 separate_namespaces: bool = False,
                 target_namespace: str = "target_tokens",
                 save_copy_fields: bool = False,
                 save_pgn_fields: bool = False,
                 lowercase: bool = True) -> None:
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

        self._cnn_tokenized_dir = cnn_tokenized_dir
        self._dm_tokenized_dir = dm_tokenized_dir

    def parse_set(self, urls_path: str) -> Iterable[Tuple[str, str]]:
        file_names = get_file_names_by_urls(self._cnn_tokenized_dir, self._dm_tokenized_dir, urls_path)
        for file_name in file_names:
            yield get_article_and_abstract(file_name)
