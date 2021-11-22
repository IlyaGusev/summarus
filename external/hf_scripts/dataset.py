import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


class SummaryDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        sample_rate: float,
        tokenizer: AutoTokenizer,
        max_source_tokens_count: int,
        max_target_tokens_count: int
    ):
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue
            tensors = self.convert_pair(
                text=record["text"],
                summary=record.get("summary")
            )
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def convert_pair(self, text, summary):
        raise NotImplementedError


class SummarySeq2SeqDataset(SummaryDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_pair(self, text, summary):
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_source_tokens_count,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        if summary is not None:
            outputs = self.tokenizer(
                summary,
                add_special_tokens=True,
                max_length=self.max_target_tokens_count,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            labels = outputs["input_ids"].squeeze(0)
            labels[outputs["attention_mask"].squeeze(0) == 0] = -100
            inputs["labels"] = labels
        return inputs


class SummaryLMDataset(SummaryDataset):
    def __init__(self, only_summary_loss=False, *args, **kwargs):
        self.only_summary_loss = only_summary_loss
        super().__init__(*args, **kwargs)

    def convert_pair(self, text, summary=None):
        text_tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_source_tokens_count,
            padding=False,
            truncation=True
        )["input_ids"]
        input_ids = text_tokens + [self.tokenizer.sep_token_id]
        if summary:
            summary_tokens = self.tokenizer(
                summary,
                add_special_tokens=False,
                max_length=self.max_target_tokens_count,
                padding=False,
                truncation=True
            )["input_ids"]
            input_ids += summary_tokens + [self.tokenizer.eos_token_id]
            max_length = self.max_source_tokens_count + self.max_target_tokens_count + 2
            padding = [self.tokenizer.pad_token_id for i in range(len(input_ids), max_length)]
            input_ids.extend(padding)
        input_ids = torch.LongTensor(input_ids)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        if self.only_summary_loss:
            for i in range(len(text_tokens) + 1):
                labels[i] = -100
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
