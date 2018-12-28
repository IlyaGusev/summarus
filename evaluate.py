import os
import logging
import argparse

from pyrouge import Rouge155
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.predictors.simple_seq2seq import SimpleSeq2SeqPredictor
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from summarus.seq2seq import Seq2Seq
from summarus.datasets.cnn_dailymail_reader import CNNDailyMailReader
from summarus.datasets.contracts_reader import ContractsReader


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


def evaluate(model_path, test_path):
    params_path = os.path.join(model_path, "config.json")

    params = Params.from_file(params_path)
    reader = DatasetReader.from_params(params.pop("reader"))

    model = Model.load(params, model_path, cuda_device=0)
    model.training = False
    print(model)
    print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    eval_dir = os.path.join(model_path, "eval")
    ref_dir = os.path.join(eval_dir, "ref")
    hyp_dir = os.path.join(eval_dir, "hyp")
    src_dir = os.path.join(eval_dir, "src")
    for d in (eval_dir, ref_dir, hyp_dir, src_dir):
        if not os.path.isdir(d):
            os.mkdir(d)

    count = 0
    predictor = SimpleSeq2SeqPredictor(model, reader)
    for article, abstract in reader.parse_files(test_path):
        if isinstance(abstract, list):
            reference_sents = abstract
        elif isinstance(abstract, basestring):
            reference_sents = abstract.split(".")
        else:
            assert False
        decoded_words = predictor.predict(article)["predicted_tokens"]
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")
            except ValueError:
                fst_period_idx = len(decoded_words)
            sent = decoded_words[:fst_period_idx + 1]
            decoded_words = decoded_words[fst_period_idx + 1:]
            decoded_sents.append(' '.join(sent))
        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        reference_sents = [make_html_safe(w) for w in reference_sents]

        if count % 100 == 0:
            print("Article: ", article)
            print("Abstract: ", reference_sents)
            print("Pred abstract: ", decoded_sents)

        ref_path = os.path.join(ref_dir, str(count) + "_reference.txt")
        hyp_path = os.path.join(hyp_dir, str(count) + "_decoded.txt")
        src_path = os.path.join(src_dir, str(count) + ".txt")
        with open(ref_path, "w", encoding="utf-8") as w:
            for idx, sent in enumerate(reference_sents):
                w.write(sent) if idx == len(reference_sents) - 1 else w.write(sent + "\n")
        with open(hyp_path, "w", encoding="utf-8") as w:
            for idx, sent in enumerate(decoded_sents):
                w.write(sent) if idx == len(decoded_sents) - 1 else w.write(sent + "\n")
        with open(src_path, "w", encoding="utf-8") as w:
            w.write(article)
        count += 1

    # r = Rouge155(rouge_dir="/home/yallen/ROUGE-1.5.5")
    # r.model_filename_pattern = '#ID#_reference.txt'
    # r.system_filename_pattern = '(\d+)_decoded.txt'
    # r.model_dir = ref_dir
    # r.system_dir = hyp_dir
    # logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    # rouge_results = r.convert_and_evaluate()
    # scores = r.output_to_dict(rouge_results)
    # rouge_log(scores)


def main(model_name,
         test_path="/data/cnn_dailymail/all_test.txt"):
    assert model_name
    models_path = "models"
    model_path = os.path.join(models_path, model_name)
    assert os.path.isdir(model_path)
    evaluate(model_path, test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--test-path')
    args = parser.parse_args()
    if not args.test_path:
        main(args.model_name)
    else:
        main(args.model_name, args.test_path)

