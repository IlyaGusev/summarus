import argparse
import re

import razdel

from summarus import *
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


def postprocess(ref, hyp, is_multiple_ref=False, detokenize_after=False, tokenize_after=True):
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
        hyp = " ".join([token.text for token in razdel.tokenize(hyp)])
        ref = " ".join([token.text for token in razdel.tokenize(ref)])
    return ref, hyp


def evaluate(predicted_path,
             gold_path,
             metric,
             max_count=None,
             is_multiple_ref=False,
             detokenize_after=False,
             tokenize_after=False,
             meteor_jar=None):
    hyps = []
    refs = []
    with open(gold_path, "r") as gold, open(predicted_path, "r") as pred:
        for i, (ref, hyp) in enumerate(zip(gold, pred)):
            if max_count is not None and i >= max_count:
                break
            ref, hyp = postprocess(ref, hyp, is_multiple_ref, detokenize_after, tokenize_after)
            refs.append(ref)
            hyps.append(hyp)
    print_metrics(refs, hyps, metric, meteor_jar)


if __name__ == "__main__":
    possible_choices = ("rouge", "bleu", "meteor", "duplicate_ngrams", "all")
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted-path', type=str, required=True)
    parser.add_argument('--gold-path', type=str, required=True)
    parser.add_argument('--metric', choices=possible_choices, default="all")
    parser.add_argument('--is-multiple-ref', action='store_true')
    parser.add_argument('--max-count', type=int, default=None)
    parser.add_argument('--detokenize-after', action='store_true')
    parser.add_argument('--tokenize-after', action='store_true')
    parser.add_argument('--meteor-jar', type=str, default=None)
    args = parser.parse_args()
    evaluate(**vars(args))
