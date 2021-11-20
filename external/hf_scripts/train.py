import argparse
import random
import json

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
from transformers import logging, T5ForConditionalGeneration


def convert_to_tensors(
    tokenizer,
    text,
    max_source_tokens_count,
    max_target_tokens_count = None,
    summary = None
):
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_source_tokens_count,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    if summary is not None:
        outputs = tokenizer(
            summary,
            add_special_tokens=True,
            max_length=max_target_tokens_count,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = outputs["input_ids"].squeeze(0)
        labels[outputs["attention_mask"].squeeze(0) == 0] = -100
        inputs["labels"] = labels
    return inputs


class SummaryDataset(Dataset):
    def __init__(
        self,
        original_records,
        sample_rate,
        tokenizer,
        max_source_tokens_count,
        max_target_tokens_count
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
            tensors = convert_to_tensors(
                tokenizer=tokenizer,
                summary=record["summary"],
                text=record["text"],
                max_target_tokens_count=self.max_target_tokens_count,
                max_source_tokens_count=self.max_source_tokens_count
            )
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def read_jsonl(file_path):
    with open(file_path) as r:
        for line in r:
            yield json.loads(line)


def train(
    config_file,
    checkpoint,
    train_file,
    val_file,
    train_sample_rate,
    val_sample_rate,
    output_dir,
    report_to,
    model_type,
    model_name
):
    logging.set_verbosity_info()
    with open(config_file, "r") as r:
        config = json.load(r)

    max_source_tokens_count = config["max_source_tokens_count"]
    max_target_tokens_count = config["max_target_tokens_count"]

    # Data preparation
    train_records = list(read_jsonl(train_file))
    random.shuffle(train_records)

    val_records = list(read_jsonl(val_file))

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, strip_accents=False)
    train_dataset = SummaryDataset(
        train_records,
        train_sample_rate,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count
    )
    val_dataset = SummaryDataset(
        val_records,
        val_sample_rate,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count
    )

    # Model loading
    if model_type == "encoder_decoder":
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
    elif model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        assert False

    # Special tokens
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model.config.pad_token_id is not None

    bos_candidates = (
        tokenizer.bos_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id
    )
    for bos_candidate in bos_candidates:
        model.config.bos_token_id = bos_candidate
        if bos_candidate is not None:
            break
    assert model.config.bos_token_id is not None
    model.config.decoder_start_token_id = model.config.bos_token_id

    eos_candidates = (tokenizer.eos_token_id, tokenizer.sep_token_id)
    for eos_candidate in eos_candidates:
        model.config.eos_token_id = eos_candidate
        if eos_candidate is not None:
            break
    assert model.config.eos_token_id is not None

    batch_size = config["batch_size"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    logging_steps = config["logging_steps"]
    eval_steps = config["eval_steps"]
    save_steps = config["save_steps"]
    learning_rate = config["learning_rate"]
    warmup_steps = config["warmup_steps"]
    num_train_epochs = config["num_train_epochs"]

    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_steps=save_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=report_to
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train(checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-sample-rate", type=float, default=1.0)
    parser.add_argument("--val-sample-rate", type=float, default=1.0)
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()
    train(**vars(args))
