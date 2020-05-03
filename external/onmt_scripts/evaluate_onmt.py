import argparse

import razdel
from summarus.util.metrics import print_metrics


def main(gold_path, predicted_path, metric, tokenize_after):
    refs = []
    hyps = []
    with open(gold_path, "r") as gold, open(predicted_path, "r") as pred:
        for gold_summary, pred_summary in zip(gold, pred):
            gold_summary = "".join(gold_summary.split(" ")[1:-1]).replace("▁", " ").strip()
            pred_summary = "".join(pred_summary.split(" ")[1:-1]).replace("▁", " ").strip()
            if tokenize_after:
                pred_summary = " ".join([token.text for token in razdel.tokenize(pred_summary)])
                pred_summary = pred_summary.replace("@ @ UNKNOWN @ @", "@@UNKNOWN@@")
                gold_summary = " ".join([token.text for token in razdel.tokenize(gold_summary)])
            refs.append(gold_summary)
            hyps.append(pred_summary)
    print_metrics(refs, hyps, metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold-path', type=str, required=True)
    parser.add_argument('--predicted-path', type=str, required=True)
    parser.add_argument('--metric', choices=("rouge", "bleu", "all"), default="all")
    parser.add_argument('--tokenize-after', action='store_true')
    args = parser.parse_args()
    main(**vars(args))
