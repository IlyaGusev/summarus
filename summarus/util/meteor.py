# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help
# Based on https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/meteor/meteor.py
# Modified by Ilya Gusev

import subprocess
import threading


class Meteor:
    def __init__(self, meteor_jar, language):
        # Used to guarantee thread safety
        self.lock = threading.Lock()

        self.meteor_cmd = ['java', '-jar', '-Xmx2G', meteor_jar, '-', '-', '-stdio', '-l', language, '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,
                                         encoding='utf-8',
                                         bufsize=0)

    def compute_score(self, hyps, refs):
        scores = []
        self.lock.acquire()
        for hyp, ref in zip(hyps, refs):
            stat = self._stat(hyp, ref)
            # EVAL ||| stats
            eval_line = 'EVAL ||| {}'.format(" ".join(map(str, map(int, map(float, stat.split())))))
            self.meteor_p.stdin.write('{}\n'.format(eval_line))
            scores.append(float(self.meteor_p.stdout.readline().strip()))
        self.lock.release()

        return sum(scores) / len(scores)

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        return self.meteor_p.stdout.readline().strip()

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()
