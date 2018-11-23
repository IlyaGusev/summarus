import hashlib
import os
from typing import Iterable, Dict

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
            file_name = os.path.join(cnn_tokenized_dir, str(hashhex(url)) + ".story")
            cnn_file_name = os.path.join(cnn_tokenized_dir, file_name)
            dm_file_name = os.path.join(dm_tokenized_dir, file_name)
            if os.path.isfile(cnn_file_name):
                yield cnn_file_name
                continue
            elif os.path.isfile(dm_file_name):
                yield dm_file_name
                continue
            assert False, "File not found in tokenized dir!"


def get_lines(file_name):
    with open(file_name, "r", encoding="utf-8") as r:
        for line in r:
            yield fix_missing_period(line.strip().lower())


def get_article_and_abstract(story_file):
    lines = get_lines(story_file)
    article_lines = []
    highlights = []
    next_is_highlight = False
    for line in lines:
        if not line:
            continue
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    article = ' '.join(article_lines)
    abstract = ' '.join(highlights)
    return article, abstract


class CNNDailyMailReader(DatasetReader):
    def __init__(self,
                 cnn_tokenized_dir: str,
                 dm_tokenized_dir: str,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)

        self.cnn_tokenized_dir = cnn_tokenized_dir
        self.dm_tokenized_dir = dm_tokenized_dir
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, urls_file_path: str) -> Iterable[Instance]:
        for article, abstract in self._parse_files(urls_file_path):
            tokenized_article = self._tokenizer.tokenize(article)
            tokenized_article.insert(0, Token(START_SYMBOL))
            tokenized_article.append(Token(END_SYMBOL))

            tokenized_abstract = self._tokenizer.tokenize(abstract)
            tokenized_abstract.insert(0, Token(START_SYMBOL))
            tokenized_abstract.append(Token(END_SYMBOL))

            article_field = TextField(tokenized_article[:1200], self._token_indexers)
            abstract_field = TextField(tokenized_abstract[:500], self._token_indexers)

            yield Instance({
                'source_tokens': article_field,
                'target_tokens': abstract_field
            })

    def text_to_instance(self, article) -> Instance:
        tokenized_article = self._tokenizer.tokenize(article)
        tokenized_article.insert(0, Token(START_SYMBOL))
        tokenized_article.append(Token(END_SYMBOL))
        article_field = TextField(tokenized_article, self._token_indexers)
        yield Instance({
            'source_tokens': article_field,
        })

    def _parse_files(self, urls_path):
        file_names = get_file_names_by_urls(self.cnn_tokenized_dir, self.dm_tokenized_dir, urls_path)
        for file_name in file_names:
            yield get_article_and_abstract(file_name)
