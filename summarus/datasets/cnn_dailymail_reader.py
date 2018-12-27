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


def get_lines(file_name):
    with open(file_name, "r", encoding="utf-8") as r:
        for line in r:
            yield fix_missing_period(line.strip().lower())


def get_article_and_abstract(story_file) -> Tuple[str, List]:
    lines = get_lines(story_file)
    article_lines = []
    abstract = []
    next_is_highlight = False
    for line in lines:
        if not line:
            continue
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            abstract.append(line)
        else:
            article_lines.append(line)

    article = ' '.join(article_lines)
    return article, abstract


@DatasetReader.register("cnn_dailymail")
class CNNDailyMailReader(DatasetReader):
    def __init__(self,
                 cnn_tokenized_dir: str = None,
                 dm_tokenized_dir: str = None,
                 tokenizer: Tokenizer = None,
                 article_token_indexers: Dict[str, TokenIndexer] = None,
                 abstract_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = True,
                 article_max_tokens: int = 400,
                 abstract_max_tokens: int = 100,
                 separate_namespaces: bool = False) -> None:
        super().__init__(lazy)

        self._cnn_tokenized_dir = cnn_tokenized_dir
        self._dm_tokenized_dir = dm_tokenized_dir

        self._article_max_tokens = article_max_tokens
        self._abstract_max_tokens = abstract_max_tokens

        self._tokenizer = tokenizer or WordTokenizer()

        tokens_indexer = {"tokens": SingleIdTokenIndexer()}
        self._article_token_indexers = article_token_indexers or tokens_indexer
        self._abstract_token_indexers = abstract_token_indexers or tokens_indexer
        if separate_namespaces:
            self._abstract_token_indexers = abstract_token_indexers or \
                                            {"tokens": SingleIdTokenIndexer(namespace="abstract_tokens")}

    def _read(self, urls_file_path: str) -> Iterable[Instance]:
        for article, abstract in self.parse_files(urls_file_path):
            if not article.strip() or not ''.join(abstract):
                continue
            instance = self.text_to_instance(article, abstract)
            yield instance

    def text_to_instance(self, article: str, abstract: List = None) -> Instance:
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

    def parse_files(self, urls_path):
        file_names = get_file_names_by_urls(self._cnn_tokenized_dir, self._dm_tokenized_dir, urls_path)
        for file_name in file_names:
            yield get_article_and_abstract(file_name)
