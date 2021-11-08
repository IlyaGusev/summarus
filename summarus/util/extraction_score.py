# Based on description from https://arxiv.org/abs/1802.01457

import math
from functools import lru_cache

import razdel


class Intervals:
    def __init__(self):
        self.intervals = []

    def try_add(self, interval):
        for i in self.intervals:
            if interval[1] >= i[0] and interval[0] <= i[0]:
                return False
            elif interval[0] >= i[0] and interval[1] <= i[1]:
                return False
            elif interval[0] <= i[1] and interval[1] >= i[1]:
                return False
        self.intervals.append(interval)
        return True


def find_acs(s1, s2, threshold=2):
    hashed_seqs = set()
    for i in range(len(s1)):
        for j in range(i+1, i+1+len(s2)):
            seq = tuple(s1[i:j])
            if len(seq) >= threshold:
                hashed_seqs.add(tuple(s1[i:j]))
    cs = []
    for i in range(len(s2)):
        for j in range(i+1, len(s2)+1):
            seq = tuple(s2[i:j])
            if len(seq) < threshold:
                continue
            if seq in hashed_seqs:
                cs.append((len(seq), seq, i, j))
    cs.sort(key=lambda x: -x[0])
    answer = []
    intervals = Intervals()
    for a, seq, i, j in cs:
        if intervals.try_add((i, j)):
            answer.append(a)
    return answer


@lru_cache(maxsize=10)
def calc_extraction_score(text, summary, threshold=2):
    text_tokens = [t.text for t in razdel.tokenize(text.lower())]
    summary_tokens = [t.text for t in razdel.tokenize(summary.lower())]
    acs = find_acs(text_tokens, summary_tokens, threshold)
    answer = 0.0
    for s in acs:
        s = s/len(summary_tokens)
        answer += s * (math.exp(s-1) - (1-s)/math.exp(1))
    return answer, acs[0] / len(summary_tokens) if acs else 0.0
