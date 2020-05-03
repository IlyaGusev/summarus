import argparse

import torch
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader
from transformers import BertTokenizer
from razdel import sentenize

from summarus.readers import *


class BertData:
    def __init__(self, bert_model, lower, max_src_tokens, max_tgt_tokens):
        self.max_src_tokens = max_src_tokens
        self.max_tgt_tokens = max_tgt_tokens
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=lower)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused1] '
        self.tgt_eos = ' [unused2]'
        self.tgt_sent_split = ' [unused3] '
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt):
        src_txt = [' '.join(s) for s in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_tokens = self.tokenizer.tokenize(text)[:self.max_src_tokens]
        src_tokens.insert(0, self.cls_token)
        src_tokens.append(self.sep_token)
        src_indices = self.tokenizer.convert_tokens_to_ids(src_tokens)

        _segs = [-1] + [i for i, t in enumerate(src_indices) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_indices) if t == self.cls_vid]

        tgt_txt = ' <q> '.join([' '.join(sentence) for sentence in tgt])
        tgt_tokens = [' '.join(self.tokenizer.tokenize(' '.join(sentence))) for sentence in tgt]
        tgt_tokens_str = self.tgt_bos + self.tgt_sent_split.join(tgt_tokens) + self.tgt_eos
        tgt_tokens = tgt_tokens_str.split()[:self.max_tgt_tokens]
        tgt_indices = self.tokenizer.convert_tokens_to_ids(tgt_tokens)

        return src_indices, tgt_indices, segments_ids, cls_ids, src_txt, tgt_txt


def preprocess(config_path, file_path, save_path, bert_path, max_src_tokens, max_tgt_tokens, lower=False, nrows=None):
    bert = BertData(bert_path, lower, max_src_tokens, max_tgt_tokens)
    params = Params.from_file(config_path)
    reader_params = params.pop("reader", default=Params({}))
    reader = DatasetReader.from_params(reader_params)
    data = []
    for i, (text, summary) in enumerate(reader.parse_set(file_path)):
        if nrows is not None and i >= nrows:
            break
        src = [(s.text.lower() if lower else s.text).split() for s in sentenize(text)]
        tgt = [(s.text.lower() if lower else s.text).split() for s in sentenize(summary)]
        src_indices, tgt_indices, segments_ids, cls_ids, src_txt, tgt_txt = bert.preprocess(src, tgt)
        b_data_dict = {
            "src": src_indices, "tgt": tgt_indices,
            "segs": segments_ids, 'clss': cls_ids,
            'src_txt': src_txt, "tgt_txt": tgt_txt
        }
        data.append(b_data_dict)
    torch.save(data, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--file-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--bert-path', type=str, required=True)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--max-src-tokens', type=int, default=600)
    parser.add_argument('--max-tgt-tokens', type=int, default=200)
    parser.add_argument('--nrows', type=int, default=None)
    args = parser.parse_args()
    preprocess(**vars(args))
