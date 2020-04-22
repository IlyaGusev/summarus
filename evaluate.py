import os
import argparse
import re
from typing import Dict
from collections import Counter

from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.predictors.seq2seq import Seq2SeqPredictor
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu
import razdel
from rouge import Rouge

from summarus import *
from summarus.util.meteor import Meteor


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
    model.eval()
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


def calc_duplicate_n_grams_rate(documents):
    all_ngrams_count = Counter()
    duplicate_ngrams_count = Counter()
    for doc in documents:
        words = doc.split(" ")
        for n in range(1, 5):
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            unique_ngrams = set(ngrams)
            all_ngrams_count[n] += len(ngrams)
            duplicate_ngrams_count[n] += len(ngrams) - len(unique_ngrams)
    return {n: duplicate_ngrams_count[n]/all_ngrams_count[n] for n in range(1, 5)}


def calc_metrics(refs, hyps, metric, meteor_jar=None):
    print("Count:", len(hyps))
    print("Ref:", refs[-1])
    print("Hyp:", hyps[-1])

    many_refs = [[r] if r is not list else r for r in refs]
    if metric in ("bleu", "all"):
        print("BLEU: ", corpus_bleu(many_refs, hyps))
    if metric == "legacy_rouge":
        print(calc_legacy_rouge(refs, hyps))
    if metric in ("rouge", "all"):
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True)
        print("ROUGE: ", scores)
    if metric in ("meteor", "all") and meteor_jar is not None and os.path.exists(meteor_jar):
        meteor = Meteor(meteor_jar, language="ru")
        print("METEOR: ", meteor.compute_score(hyps, many_refs))
    if metric in ("duplicate_bigrams", "all"):
        print("Duplicate bigrams: ", calc_duplicate_n_grams_rate(hyps)[2] * 100)


def evaluate(test_path, batch_size, metric,
             max_count, report_every, is_multiple_ref=False,
             model_path=None, model_config_path=None, baseline=None,
             reader_config_path=None, detokenize_after=False,
             tokenize_after=False, meteor_jar=None):
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
            if tokenize_after:
                hyp = " ".join([token.text for token in razdel.tokenize(hyp)])
                hyp = hyp.replace("@ @ UNKNOWN @ @", "@@UNKNOWN@@")
                ref = " ".join([token.text for token in razdel.tokenize(ref)])
            if isinstance(ref, str) and len(ref) <= 1:
                ref = "some content"
                print("Empty ref")
            if isinstance(hyp, str) and len(hyp) <= 1:
                hyp = "some content"
                print("Empty hyp. Ref: ", ref)

            refs.append(ref)
            hyps.append(hyp)
            if len(hyps) % report_every == 0:
                calc_metrics(refs, hyps, metric, meteor_jar)
            if max_count and len(hyps) >= max_count:
                break
    calc_metrics(refs, hyps, metric, meteor_jar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--test-path', type=str, required=True)
    parser.add_argument('--model-config-path', type=str, default=None)
    parser.add_argument('--reader-config-path', type=str, default=None)
    parser.add_argument('--baseline', choices=("lead1", "lead1skip1", "lead2", "lead3",
                                               "lead4", "lead5", "lead6"), default=None)
    parser.add_argument('--metric', choices=("rouge", "bleu", "meteor", "all"), default="all")
    parser.add_argument('--is-multiple-ref', action='store_true')
    parser.add_argument('--max-count', type=int, default=None)
    parser.add_argument('--report-every', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--detokenize-after', action='store_true')
    parser.add_argument('--tokenize-after', action='store_true')
    parser.add_argument('--meteor-jar', type=str, default=None)

    args = parser.parse_args()
    assert os.path.isdir(args.model_path)
    evaluate(**vars(args))
