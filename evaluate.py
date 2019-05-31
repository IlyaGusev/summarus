import os
import argparse
import re
from typing import Dict

from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.predictors.seq2seq import Seq2SeqPredictor
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
import torch
import nltk
from rouge import Rouge

from summarus import *


def detokenize(text):
    text = text.strip()
    punctuation = ",.!?:;%"
    closing_punctuation = ")]}"
    opening_punctuation = "([}"
    for ch in punctuation + closing_punctuation:
        text = text.replace(" " + ch, ch)
    for ch in opening_punctuation:
        text = text.replace(ch + " ", ch)
    res = [r'"\s[^"]+\s"', r"'\s[^']+\s'"]
    for r in res:
        for f in re.findall(r, text, re.U):
            text = text.replace(f, f[0] + f[2:-2] + f[-1])
    text = text.replace("' s", "'s").replace(" 's", "'s")
    text = text.strip()
    return text


def get_batches(reader: SummarizationReader, test_path: str, batch_size: int) -> Dict:
    batch = []
    for source, target in reader.parse_set(test_path):
        source = source.strip()
        batch.append({"source": source, "target": target})
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_baseline(batch, baseline):
    sources = [b.get('source') for b in batch]
    targets = [b.get('target') for b in batch]
    hyps = []
    for source in sources:
        source_sentences = nltk.sent_tokenize(source)
        if baseline.startswith("lead"):
            skip = 0
            if "skip" in baseline:
                skip = int(baseline.split("skip")[-1])
                n = int(baseline.split("skip")[0].replace("lead", ""))
            else:
                n = int(baseline.replace("lead", ""))
            if len(source_sentences) == 1:
                skip = 0
            hyp = " ".join(source_sentences[skip:skip+n]).strip()
        else:
            assert False
        hyps.append(hyp)
    return targets, hyps


def get_model_runner(model_path, reader, model_config_path=None):
    config_path = model_config_path or os.path.join(model_path, "config.json")
    params = Params.from_file(config_path)
    device = 0 if torch.cuda.is_available() else -1
    model = Model.load(params, model_path, cuda_device=device)
    model.training = False
    predictor = Seq2SeqPredictor(model, reader)
    print(model)
    print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    def run_model(batch):
        outputs = predictor.predict_batch_json(batch)
        targets = [b.get('target') for b in batch]
        hyps = []
        for output in outputs:
            decoded_words = output["predicted_tokens"]
            hyp = " ".join(decoded_words).strip()
            hyps.append(hyp)
        return targets, hyps
    return run_model


def get_reader_params(reader_config_path=None, model_config_path=None, model_path=None):
    assert reader_config_path or model_config_path or model_path
    if reader_config_path:
        reader_params = Params.from_file(reader_config_path)
    else:
        reader_params_path = model_config_path or os.path.join(model_path, "config.json")
        reader_params = Params.from_file(reader_params_path).pop("reader")
    return reader_params


def calc_legacy_rouge(refs, hyps, directory="eval"):
    from pyrouge import Rouge155
    r = Rouge155()
    system_dir = os.path.join(directory, 'hyp')
    model_dir = os.path.join(directory, 'ref')
    if not os.path.isdir(system_dir):
        os.makedirs(system_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    r.system_dir = system_dir
    r.model_dir = model_dir
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_filename_pattern = '#ID#_reference.txt'
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        hyp_file_path = os.path.join(r.system_dir, "%06d_decoded.txt" % i)
        with open(hyp_file_path, "w") as w:
            hyp_sentences = hyp.split(" s_s ")
            w.write("\n".join(hyp_sentences))
        ref_file_path = os.path.join(r.model_dir, "%06d_reference.txt" % i)
        with open(ref_file_path, "w") as w:
            ref_sentences = ref.split(" s_s ")
            w.write("\n".join(ref_sentences))
    output = r.convert_and_evaluate()
    result = r.output_to_dict(output)
    log_str = ""
    for x in ["1","2","l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x,y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = result[key]
            val_cb = result[key_cb]
            val_ce = result[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    return log_str


def calc_metrics(refs, hyps, metric):
    print("Count:", len(hyps))
    print("Ref:", refs[-1])
    print("Hyp:", hyps[-1])

    if metric in ("bleu", "all"):
        from nltk.translate.bleu_score import corpus_bleu
        print("BLEU: ", corpus_bleu([[r] if r is not list else r for r in refs], hyps))
    if metric == "legacy_rouge":
        print(calc_legacy_rouge(refs, hyps))
    if metric in ("rouge", "all"):
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True)
        print("ROUGE: ", scores)


def evaluate(test_path, batch_size, metric,
             max_count, report_every, is_multiple_ref=False,
             model_path=None, model_config_path=None, baseline=None,
             reader_config_path=None, detokenize_after=False):
    reader_params = get_reader_params(reader_config_path, model_config_path, model_path)
    is_subwords = "tokenizer" in reader_params and reader_params["tokenizer"]["type"] == "subword"
    reader = DatasetReader.from_params(reader_params)
    run_model = get_model_runner(model_path, reader) if not baseline else None

    hyps = []
    refs = []
    for batch in get_batches(reader, test_path, batch_size):
        batch_refs, batch_hyps = run_model(batch) if not baseline else run_baseline(batch, baseline)
        for ref, hyp in zip(batch_refs, batch_hyps):
            hyp = hyp if not is_subwords else "".join(hyp.split(" ")).replace("▁", " ")
            if is_multiple_ref:
                reference_sents = ref.split(" s_s ")
                decoded_sents = hyp.split("s_s")
                hyp = [w.replace("<", "&lt;").replace(">", "&gt;").strip() for w in decoded_sents]
                ref = [w.replace("<", "&lt;").replace(">", "&gt;").strip() for w in reference_sents]
                hyp = " ".join(hyp)
                ref = " ".join(ref)
            ref = ref.strip()
            hyp = hyp.strip()
            if detokenize_after:
                hyp = detokenize(hyp)
                ref = detokenize(ref)
            if isinstance(ref, str) and len(ref) <= 1:
                ref = "some content"
                print("Empty ref")
            if isinstance(hyp, str) and len(hyp) <= 1:
                hyp = "some content"
                print("Empty hyp. Ref: ", ref)

            refs.append(ref)
            hyps.append(hyp)
            if len(hyps) % report_every == 0:
                calc_metrics(refs, hyps, metric)
            if max_count and len(hyps) >= max_count:
                break
    calc_metrics(refs, hyps, metric)


def main(**kwargs):
    assert os.path.isdir(kwargs['model_path'])
    evaluate(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--test-path', type=str, required=True)
    parser.add_argument('--model-config-path', type=str, default=None)
    parser.add_argument('--reader-config-path', type=str, default=None)
    parser.add_argument('--baseline', choices=("lead1", "lead1skip1", "lead2", "lead3",
                                               "lead4", "lead5", "lead6"), default=None)
    parser.add_argument('--metric', choices=("rouge", "legacy_rouge", "bleu", "all"), default="all")
    parser.add_argument('--is-multiple-ref', action='store_true')
    parser.add_argument('--max-count', type=int, default=None)
    parser.add_argument('--report-every', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--detokenize-after', action='store_true')

    args = parser.parse_args()
    main(**vars(args))
