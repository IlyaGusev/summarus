import os
import logging
import argparse

from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.predictors.simple_seq2seq import SimpleSeq2SeqPredictor
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from summarus.seq2seq import Seq2Seq
from summarus.readers.cnn_dailymail_reader import CNNDailyMailReader
from summarus.readers.contracts_reader import ContractsReader


def make_html_safe(s):
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def rouge_log(results_dict):
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x,y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str)


def evaluate(model_path, test_path, config_path, metric, is_multiple_ref, max_count, report_every):
    params_path = config_path or os.path.join(model_path, "config.json")

    params = Params.from_file(params_path)
    is_subwords = "tokenizer" in params["reader"] and params["reader"]["tokenizer"]["type"] == "subword"
    reader = DatasetReader.from_params(params.pop("reader"))

    model = Model.load(params, model_path, cuda_device=0)
    model.training = False
    print(model)
    print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    hyps = []
    refs = []
    predictor = SimpleSeq2SeqPredictor(model, reader)
    for source, target in reader.parse_set(test_path):
        decoded_words = predictor.predict(source)["predicted_tokens"]
        if is_multiple_ref:
            if isinstance(target, list):
                reference_sents = target
            elif isinstance(target, str):
                reference_sents = target.split(".")
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
            print("Source: ", source)
            print("Ref: ", ref)
            print("Hyp: ", hyp)

        if max_count and len(hyps) >= max_count:
            break

    if metric == "bleu":
        from nltk.translate.bleu_score import corpus_bleu
        print("BLEU: ", corpus_bleu(refs, hyps))

    if metric == "rouge":
        from pyrouge import Rouge155
        eval_dir = os.path.join(model_path, "eval")
        ref_dir = os.path.join(eval_dir, "ref")
        hyp_dir = os.path.join(eval_dir, "hyp")
        for d in (eval_dir, ref_dir, hyp_dir):
            if not os.path.isdir(d):
                os.mkdir(d)

        for ref, hyp in zip(refs, hyps):
            ref_path = os.path.join(ref_dir, str(count) + "_reference.txt")
            hyp_path = os.path.join(hyp_dir, str(count) + "_decoded.txt")
            with open(ref_path, "w", encoding="utf-8") as w:
                for idx, sent in enumerate(reference_sents):
                    w.write(sent) if idx == len(reference_sents) - 1 else w.write(sent + "\n")
            with open(hyp_path, "w", encoding="utf-8") as w:
                for idx, sent in enumerate(decoded_sents):
                    w.write(sent) if idx == len(decoded_sents) - 1 else w.write(sent + "\n")

        r = Rouge155(rouge_dir="/home/yallen/ROUGE-1.5.5")
        r.model_filename_pattern = '#ID#_reference.txt'
        r.system_filename_pattern = '(\d+)_decoded.txt'
        r.model_dir = ref_dir
        r.system_dir = hyp_dir
        logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
        rouge_results = r.convert_and_evaluate()
        scores = r.output_to_dict(rouge_results)
        rouge_log(scores)


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
