from typing import List

from sentencepiece import SentencePieceProcessor as sp_processor
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

@Tokenizer.register("subword")
class SubwordTokenizer(Tokenizer):
    def __init__(self, model_path: str="bpe.model"):
        self._model_path = model_path
        self._processor = sp_processor()
        self._processor.Load(model_path)

    def tokenize(self, text: str) -> List[Token]:
        subwords = self._processor.EncodeAsPieces(text)
        return subwords

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]
