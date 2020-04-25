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
import razdel

from summarus import *
from summarus.util.metrics import print_metrics
from summarus.predictors.sentences_tagger_predictor import SentencesTaggerPredictor


def punct_detokenize(text):
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


def get_abs_batches(reader: SummarizationReader,
                    test_path: str,
                    batch_size: int,
                    lowercase: bool = True) -> Dict:
    batch = []
    for source, target in reader.parse_set(test_path):
        source = source.strip()
        target = target.strip()
        if lowercase:
            source = source.lower() if lowercase else source
            target = target.lower() if lowercase else target
        sample = {"source": source, "target": target}
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def get_ext_batches(reader: SummarizationSentencesTaggerReader,
                    test_path: str,
                    batch_size: int,
                    lowercase: bool = True) -> Dict:
    batch = []
    for _, summary, sentences, tags in reader.parse_set(test_path):
        if lowercase:
            sentences = [sentence.lower() for sentence in sentences]
            summary = summary.lower()
        batch.append({"source_sentences": sentences, "sentences_tags": tags, "target": summary})
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_baseline(batch, baseline):
    sources = [b.get("source") for b in batch]
    targets = [b.get("target") for b in batch]
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


def run_abs_model(predictor, batch):
    outputs = predictor.predict_batch_json(batch)
    targets = [b.get("target") for b in batch]
    hyps = []
    for output in outputs:
        decoded_words = output["predicted_tokens"]
        hyp = " ".join(decoded_words).strip()
        hyps.append(hyp)
    return targets, hyps


def run_ext_model(predictor, batch, top_n=3, border=None):
    outputs = predictor.predict_batch_json(batch)
    targets = [b.get("target") for b in batch]
    hyps = []
    for sample, output in zip(batch, outputs):
        sentences = sample["source_sentences"]
        proba = output["predicted_tags"]
        if top_n is not None:
            indices = [i for i, p, in sorted(enumerate(proba), key=lambda x: -x[1])[:top_n]]
            indices.sort()
            hyp = [sentences[i] for i in indices]
        elif border is not None:
            predicted_tags = [prob > border for prob in proba]
            if sum(predicted_tags) == 0:
                best_proba = max(proba)
                predicted_tags = [p == best_proba for i, p in enumerate(proba)]
                hyp = [sentence for sentence, tag in zip(sentences, predicted_tags) if tag == 1]
        hyp = " ".join(hyp).strip()
        hyps.append(hyp)
    return targets, hyps


def postprocess(ref, hyp, is_subwords=True, is_multiple_ref=False, detokenize_after=False, tokenize_after=True):
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
        hyp = punct_detokenize(hyp)
        ref = punct_detokenize(ref)
    if tokenize_after:
        hyp = " ".join([token.text for token in razdel.tokenize(hyp)])
        hyp = hyp.replace("@ @ UNKNOWN @ @", "@@UNKNOWN@@")
        ref = " ".join([token.text for token in razdel.tokenize(ref)])
    return ref, hyp


def evaluate(test_path, batch_size, metric, mode,
             max_count, report_every, is_multiple_ref=False,
             model_path=None, model_config_path=None,
             detokenize_after=False, tokenize_after=False, meteor_jar=None):
    config_path = model_config_path or os.path.join(model_path, "config.json")
    params = Params.from_file(config_path)
    reader_params = params["reader"].duplicate()
    device = 0 if torch.cuda.is_available() else -1
    is_subwords = "tokenizer" in reader_params and reader_params["tokenizer"]["type"] == "subword"
    reader = DatasetReader.from_params(reader_params)
    hyps = []
    refs = []

    if mode == "abs":
        model = Model.load(params, model_path, cuda_device=device)
        model.eval()
        predictor = Seq2SeqPredictor(model, reader)
        print(model)
        print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    elif mode == "ext":
        model = Model.load(params, model_path, cuda_device=device)
        model.eval()
        predictor = SentencesTaggerPredictor(model, reader)
        print(model)
        print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    iter_batches = get_ext_batches if mode == "ext" else get_abs_batches
    for batch in iter_batches(reader, test_path, batch_size):
        if mode == "abs":
            batch_refs, batch_hyps = run_abs_model(predictor, batch)
        elif mode == "ext":
            batch_refs, batch_hyps = run_ext_model(predictor, batch)
        else:
            batch_refs, batch_hyps = run_baseline(batch, mode)
        for ref, hyp in zip(batch_refs, batch_hyps):
            ref, hyp = postprocess(ref, hyp, is_subwords if mode != 'ext' else False, is_multiple_ref, detokenize_after, tokenize_after)
            refs.append(ref)
            hyps.append(hyp)
            if len(hyps) % report_every == 0:
                print_metrics(refs, hyps, metric, meteor_jar)
            if max_count and len(hyps) >= max_count:
                break
    print_metrics(refs, hyps, metric, meteor_jar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--test-path', type=str, required=True)
    parser.add_argument('--model-config-path', type=str, default=None)
    parser.add_argument('--mode', choices=("lead1", "lead1skip1", "lead2", "lead3",
                                            "lead4", "lead5", "lead6", "abs", "ext"), default="abs")
    parser.add_argument('--metric', choices=("rouge", "bleu", "meteor", "duplicate_ngrams", "all"), default="all")
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
