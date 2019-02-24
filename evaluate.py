import os
import logging
import argparse

from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.predictors.seq2seq import Seq2SeqPredictor
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from rouge import Rouge

from summarus import *


def make_html_safe(s):
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def evaluate(model_path, test_path, config_path, metric, is_multiple_ref, max_count, report_every):
    params_path = config_path or os.path.join(model_path, "config.json")

    params = Params.from_file(params_path)
    is_subwords = "tokenizer" in params["reader"] and params["reader"]["tokenizer"]["type"] == "subword"
    reader = DatasetReader.from_params(params.pop("reader"))

    device = 0 if torch.cuda.is_available() else -1
    model = Model.load(params, model_path, cuda_device=device)
    model.training = False
    print(model)
    print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    hyps = []
    refs = []
    predictor = Seq2SeqPredictor(model, reader)
    for source, target in reader.parse_set(test_path):
        decoded_words = predictor.predict(source)["predicted_tokens"]
        if is_multiple_ref:
            if isinstance(target, list):
                reference_sents = target
            elif isinstance(target, str):
                reference_sents = target.split(" s_s ")
            else:
                assert False
            decoded_sents = []
            while len(decoded_words) > 0:
                try:
                    fst_period_idx = decoded_words.index(".")
                except ValueError:
                    fst_period_idx = len(decoded_words)
                sent = decoded_words[:fst_period_idx + 1]
                decoded_words = decoded_words[fst_period_idx + 1:]
                decoded_sents.append(' '.join(sent))
            hyp = [make_html_safe(w) for w in decoded_sents]
            ref = [make_html_safe(w) for w in reference_sents]
        else:
            if not is_subwords:
                hyp = " ".join(decoded_words)
            else:
                hyp = "".join(decoded_words).replace("▁", " ") 
            ref = [target]

        hyps.append(hyp)
        refs.append(ref)

        if len(hyps) % report_every == 0:
            print("Count: ", len(hyps))
            print("Ref: ", ref)
            print("Hyp: ", hyp)
            if metric == "bleu":
                from nltk.translate.bleu_score import corpus_bleu
                print("BLEU: ", corpus_bleu(refs, hyps))

            if metric == "rouge":
                rouge = Rouge()
                scores = rouge.get_scores(hyps, [r[0] for r in refs], avg=True)
                print("ROUGE: ", scores)

        if max_count and len(hyps) >= max_count:
            break


def main(**kwargs):
    assert os.path.isdir(kwargs['model_path'])
    evaluate(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--test-path', required=True)
    parser.add_argument('--config-path', default=None)
    parser.add_argument('--metric', choices=("rouge", "bleu"))
    parser.add_argument('--is-multiple-ref', dest='is_multiple_ref', action='store_true')
    parser.add_argument('--max-count', type=int, default=None)
    parser.add_argument('--report-every', type=int, default=100)
    parser.set_defaults(is_multiple_ref=False)

    args = parser.parse_args()
    main(**vars(args))
