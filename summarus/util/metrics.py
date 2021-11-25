import os
from collections import Counter
from statistics import mean

from true_rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf
import torch

from summarus.util.meteor import Meteor


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
    return {n: duplicate_ngrams_count[n]/all_ngrams_count[n] if all_ngrams_count[n] else 0.0
            for n in range(1, 5)}


def calc_bert_score(
    hyps,
    refs,
    lang="ru",
    bert_score_model=None,
    num_layers=None,
    idf=False,
    batch_size=32
):
    import bert_score
    all_preds, hash_code = bert_score.score(
        hyps,
        refs,
        lang=lang,
        model_type=bert_score_model,
        num_layers=num_layers,
        verbose=False,
        idf=idf,
        batch_size=batch_size,
        return_hash=True
    )
    avg_scores = [s.mean(dim=0) for s in all_preds]
    return {
        "p": avg_scores[0].cpu().item(),
        "r": avg_scores[1].cpu().item(),
        "f": avg_scores[2].cpu().item()
    }, hash_code


def calc_metrics(
    refs, hyps,
    language,
    metric="all",
    meteor_jar=None
):
    metrics = dict()
    metrics["count"] = len(hyps)
    metrics["ref_example"] = refs[-1]
    metrics["hyp_example"] = hyps[-1]
    many_refs = [[r] if r is not list else r for r in refs]
    if metric in ("bleu", "all"):
        t_hyps = [hyp.split(" ") for hyp in hyps]
        t_refs = [[r.split(" ") for r in rs] for rs in many_refs]
        metrics["bleu"] = corpus_bleu(t_refs, t_hyps)
    if metric in ("rouge", "all"):
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True)
        metrics.update(scores)
    if metric in ("meteor", "all") and meteor_jar is not None and os.path.exists(meteor_jar):
        meteor = Meteor(meteor_jar, language=language)
        metrics["meteor"] = meteor.compute_score(hyps, many_refs)
    if metric in ("duplicate_ngrams", "all"):
        metrics["duplicate_ngrams"] = dict()
        metrics["duplicate_ngrams"].update(calc_duplicate_n_grams_rate(hyps))
    if metric in ("bert_score",) and torch.cuda.is_available():
        bert_scores, hash_code = calc_bert_score(hyps, refs)
        metrics["bert_score_{}".format(hash_code)] = bert_scores
    if metric in ("chrf", "all"):
        metrics["chrf"] = corpus_chrf(refs, hyps, beta=1.0)
    if metric in ("length", "all"):
        metrics["length"] = mean([len(h) for h in hyps])
    return metrics


def print_metrics(refs, hyps, language, metric="all", meteor_jar=None):
    metrics = calc_metrics(refs, hyps, language=language, metric=metric, meteor_jar=meteor_jar)

    print("-------------METRICS-------------")
    print("Count:\t", metrics["count"])
    print("Ref:\t", metrics["ref_example"])
    print("Hyp:\t", metrics["hyp_example"])

    if "bleu" in metrics:
        print("BLEU:     \t{:3.1f}".format(metrics["bleu"] * 100.0))
    if "chrf" in metrics:
        print("chrF:     \t{:3.1f}".format(metrics["chrf"] * 100.0))
    if "rouge-1" in metrics:
        print("ROUGE-1-F:\t{:3.1f}".format(metrics["rouge-1"]['f'] * 100.0))
        print("ROUGE-2-F:\t{:3.1f}".format(metrics["rouge-2"]['f'] * 100.0))
        print("ROUGE-L-F:\t{:3.1f}".format(metrics["rouge-l"]['f'] * 100.0))
    if "meteor" in metrics:
        print("METEOR:   \t{:3.1f}".format(metrics["meteor"] * 100.0))
    if "duplicate_ngrams" in metrics:
        print("Dup 1-grams:\t{:3.1f}".format(metrics["duplicate_ngrams"][1] * 100.0))
        print("Dup 2-grams:\t{:3.1f}".format(metrics["duplicate_ngrams"][2] * 100.0))
        print("Dup 3-grams:\t{:3.1f}".format(metrics["duplicate_ngrams"][3] * 100.0))
    if "length" in metrics:
        print("Avg length:\t{:3.1f}".format(metrics["length"]))
    for key, value in metrics.items():
        if "bert_score" not in key:
            continue
        print("{}:\t{:3.1f}".format(key, value["f"] * 100.0))
