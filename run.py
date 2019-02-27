import os
import argparse

import torch
from bs4 import BeautifulSoup
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.predictors.seq2seq import Seq2SeqPredictor
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from summarus import *


def get_batches(test_path, batch_size):
    with open(test_path, "r", encoding="utf-8") as f:
        batch = []
        for source in f:
            source = source.strip().lower()
            source = BeautifulSoup(source, 'html.parser').text[:15000]
            if len(source) <= 3:
                source = "риа новости"
            batch.append({"source": source})
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def run(model_path, test_path, config_path, output_path, batch_size):
    params_path = config_path or os.path.join(model_path, "config.json")

    params = Params.from_file(params_path)
    is_subwords = "tokenizer" in params["reader"] and params["reader"]["tokenizer"]["type"] == "subword"
    reader = DatasetReader.from_params(params.pop("reader"))

    device = 0 if torch.cuda.is_available() else -1
    model = Model.load(params, model_path, cuda_device=device)
    model.training = False

    predictor = Seq2SeqPredictor(model, reader)
    with open(output_path, "wt", encoding="utf-8") as w:
        for batch_number, batch in enumerate(get_batches(test_path, batch_size)):
            outputs = predictor.predict_batch_json(batch)
            assert len(outputs) == len(batch)
            for output in outputs:
                decoded_words = output["predicted_tokens"]
                if not decoded_words:
                    decoded_words = ["заявил"]
                if not is_subwords:
                    hyp = " ".join(decoded_words)
                else:
                    hyp = "".join(decoded_words).replace("▁", " ").replace("\n", "").strip()
                if len(hyp) <= 3:
                    hyp = "заявил"
                w.write(hyp + "\n")


def main(**kwargs):
    assert os.path.isdir(kwargs['model_path'])
    run(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default="models/ria_sw_cn_small")
    parser.add_argument('--test-path', default="/input.txt")
    parser.add_argument('--config-path', default=None)
    parser.add_argument('--output-path', default="/output.txt")
    parser.add_argument('--batch-size', type=int, default=1024)

    args = parser.parse_args()
    main(**vars(args))
