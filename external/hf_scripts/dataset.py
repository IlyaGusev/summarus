import random
from typing import List, Dict

import torch
import torch.nn.functional as F
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
        max_target_tokens_count: int,
        source_field: str = "text",
        target_field: str = "summary"
    ):
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count
        self.source_field = source_field
        self.target_field = target_field

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue
            tensors = self.convert_pair(
                text=record[source_field],
                summary=record.get(target_field)
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


class SummaryExtractiveDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        sample_rate: float,
        tokenizer: AutoTokenizer,
        max_source_tokens_count: int,
        max_source_sentences_count: int = None,
        use_token_level: bool = False,
        tokens_label: int = -100,
        source_field: str = "sentences",
        target_field: str = "oracle"
    ):
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_source_sentences_count = max_source_sentences_count
        self.use_token_level = use_token_level
        self.tokens_label = tokens_label

        self.records = []
        for record in tqdm(original_records):
            sentences = record[source_field]
            oracle_labels = record.get(target_field)
            if random.random() > self.sample_rate:
                continue
            tensors = self.convert(sentences, oracle_labels)
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def convert(self, sentences, labels):
        text = self.tokenizer.sep_token.join(sentences)
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_source_tokens_count,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        sep_token_id = self.tokenizer.sep_token_id
        input_ids = inputs["input_ids"]

        # Fix token_type_ids
        current_token_type_id = 0
        inputs["token_type_ids"] = input_ids.new_zeros(input_ids.size())
        for pos, input_id in enumerate(input_ids):
            inputs["token_type_ids"][pos] = current_token_type_id
            if input_id == sep_token_id:
                current_token_type_id = 1 - current_token_type_id

        # Get labels
        if labels is not None:
            if not self.use_token_level:
                labels = labels[:self.max_source_sentences_count]
                padding_size = self.max_source_sentences_count - len(labels)
                labels_tensor = torch.LongTensor(labels)
                labels_tensor = F.pad(labels_tensor, pad=(0, padding_size), mode="constant", value=-100)
            else:
                indices = [index for index, token_id in enumerate(input_ids) if token_id == sep_token_id]
                labels_tensor = input_ids.new_full(input_ids.size(), self.tokens_label)
                for index, label in zip(indices, labels):
                    labels_tensor[index] = label
            inputs["labels"] = labels_tensor
        return inputs
