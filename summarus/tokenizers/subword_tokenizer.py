from typing import List

from sentencepiece import SentencePieceProcessor
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.common.file_utils import cached_path


@Tokenizer.register("subword")
class SubwordTokenizer(Tokenizer):
    def __init__(self,
                 model_path: str = None,
                 nbest_size: int = None,
                 alpha: float = None):
        self._model_path = cached_path(model_path)
        self._processor = SentencePieceProcessor()
        self._processor.Load(self._model_path)
        self._nbest_size = nbest_size
        self._alpha = alpha

    def tokenize(self, text: str) -> List[Token]:
        if self._nbest_size and self._alpha:
            subwords = self._processor.SampleEncodeAsPieces(text, self._nbest_size, self._alpha)
        else:
            subwords = self._processor.EncodeAsPieces(text)
        tokens = [Token(s) for s in subwords]
        return tokens

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]
