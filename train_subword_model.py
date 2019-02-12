import os
import json
import tempfile
import argparse

from bs4 import BeautifulSoup
from sentencepiece import SentencePieceTrainer as sp_trainer


def parse_ria_json(path):
    with open(path, "r", encoding="utf-8") as r:
        for line in r:
            data = json.loads(line.strip())
            title = data["title"]
            text = data["text"]
            clean_text = BeautifulSoup(text, 'html.parser').text
            if not clean_text or not title:
                continue
            yield clean_text, title
   

def train_subwords(train_path, model_path, model_type, vocab_size):
    temp = tempfile.NamedTemporaryFile(mode="w", delete=False)
    for text, title in parse_ria_json(train_path):
        temp.write(text + "\n")
        temp.write(title + "\n")
    temp.close()
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    cmd = "--input={} --model_prefix={} --vocab_size={} --model_type={}".format(
        temp.name,
        os.path.join(model_path, model_type),
        vocab_size,
        model_type)
    sp_trainer.Train(cmd)
    os.unlink(temp.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--model-type', type=str, default="bpe")
    parser.add_argument('--vocab-size', type=int, default=50000)
    args = parser.parse_args()
    train_subwords(**vars(args))
