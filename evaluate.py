import os
import argparse
import re

import razdel
import nltk

from summarus.util.metrics import print_metrics


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


def postprocess(ref, hyp, language, is_multiple_ref=False, detokenize_after=False, tokenize_after=False, lower=False):
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
        hyp = hyp.replace("@@UNKNOWN@@", "<unk>")
        if language == "ru":
            hyp = " ".join([token.text for token in razdel.tokenize(hyp)])
            ref = " ".join([token.text for token in razdel.tokenize(ref)])
        else:
            hyp = " ".join([token for token in nltk.word_tokenize(hyp)])
            ref = " ".join([token for token in nltk.word_tokenize(ref)])
    if lower:
        hyp = hyp.lower()
        ref = ref.lower()
    return ref, hyp


def evaluate(predicted_path,
             gold_path,
             metric,
             language,
             max_count=None,
             is_multiple_ref=False,
             detokenize_after=False,
             tokenize_after=False,
             lower=False,
             meteor_jar=None):
    assert os.path.exists(gold_path)
    assert os.path.exists(predicted_path)
    if max_count is None:
        with open(gold_path) as gold:
            gold_num_lines = sum(1 for line in gold)
        with open(predicted_path) as pred:
            pred_num_lines = sum(1 for line in pred)
        msg = "Number of lines in files differ: {} vs {}".format(gold_num_lines, pred_num_lines)
        assert gold_num_lines == pred_num_lines, msg

    hyps = []
    refs = []
    with open(gold_path, "r") as gold, open(predicted_path, "r") as pred:
        for i, (ref, hyp) in enumerate(zip(gold, pred)):
            if max_count is not None and i >= max_count:
                break
            ref, hyp = postprocess(ref, hyp, language, is_multiple_ref, detokenize_after, tokenize_after, lower)
            if not hyp:
                print("Empty hyp for ref: ", ref)
                continue
            if not ref:
                continue
            refs.append(ref)
            hyps.append(hyp)
    print_metrics(refs, hyps, metric=metric, meteor_jar=meteor_jar, language=language)


if __name__ == "__main__":
    possible_choices = ("rouge", "bleu", "meteor", "duplicate_ngrams", "all", "bert_score", "chrf")
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted-path', type=str, required=True)
    parser.add_argument('--gold-path', type=str, required=True)
    parser.add_argument('--metric', choices=possible_choices, default="all")
    parser.add_argument('--language', choices=("ru", "en"), required=True)
    parser.add_argument('--is-multiple-ref', action='store_true')
    parser.add_argument('--max-count', type=int, default=None)
    parser.add_argument('--detokenize-after', action='store_true')
    parser.add_argument('--tokenize-after', action='store_true')
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--meteor-jar', type=str, default=None)
    args = parser.parse_args()
    evaluate(**vars(args))
