from typing import List

import razdel
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("razdel")
class RazdelTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = False):
        self._lowercase = lowercase

    def tokenize(self, text: str) -> List[Token]:
        return [Token(token.text.lower()) if self._lowercase else Token(token.text) for token in razdel.tokenize(text)]

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]
