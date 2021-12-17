import argparse
import random
import json

import torch.nn as nn
from transformers import AutoTokenizer, Trainer, TrainingArguments, logging
from transformers import BertConfig, AutoModelForTokenClassification

from extractive_model import ModelForSentencesClassification, ModelForSentencesClassificationConfig
from dataset import SummaryExtractiveDataset
from util import read_jsonl, set_random_seed, fix_tokenizer


def train(
    config_file,
    checkpoint,
    train_file,
    val_file,
    train_sample_rate,
    val_sample_rate,
    output_dir,
    report_to,
    seed,
    source_field,
    target_field
):
    set_random_seed(seed)
    logging.set_verbosity_info()
    with open(config_file, "r") as r:
        config = json.load(r)

    tokens_model_name = config["tokens_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokens_model_name)
    tokenizer = fix_tokenizer(tokenizer)

    # Data preparation
    train_records = list(read_jsonl(train_file))
    val_records = list(read_jsonl(val_file))
    random.shuffle(train_records)

    dataset_class = SummaryExtractiveDataset
    max_source_tokens_count = config["max_source_tokens_count"]
    max_source_sentences_count = config.get("max_source_sentences_count")
    tokens_label = config.get("tokens_label", -100)
    is_token_level = "sentences_model" not in config
    train_dataset_args = {
        "original_records": train_records,
        "sample_rate": train_sample_rate,
        "tokenizer": tokenizer,
        "max_source_tokens_count": max_source_tokens_count,
        "max_source_sentences_count": max_source_sentences_count,
        "use_token_level": is_token_level,
        "tokens_label": tokens_label,
        "source_field": source_field,
        "target_field": target_field
    }
    val_dataset_args = {
        "original_records": val_records,
        "sample_rate": val_sample_rate,
        "tokenizer": tokenizer,
        "max_source_tokens_count": max_source_tokens_count,
        "max_source_sentences_count": max_source_sentences_count,
        "use_token_level": is_token_level,
        "tokens_label": tokens_label,
        "source_field": source_field,
        "target_field": target_field
    }
    train_dataset = dataset_class(**train_dataset_args)
    val_dataset = dataset_class(**val_dataset_args)

    # Model loading
    if is_token_level:
        num_labels = config["num_labels"]
        model = AutoModelForTokenClassification.from_pretrained(
            tokens_model_name, num_labels=num_labels
        )
        tokens_model = model
    else:
        sentences_model_config = BertConfig(**config["sentences_model"])
        model = ModelForSentencesClassification.from_parts_pretrained(
            tokens_model_name=tokens_model_name,
            sentences_model_config=sentences_model_config
        )
        tokens_model = model.tokens_model

    # Fix model
    model.config.sep_token_id = tokenizer.sep_token_id
    model.config.max_sentences_count = max_source_sentences_count
    assert model.config.sep_token_id is not None

    # Training
    batch_size = config["batch_size"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    logging_steps = config["logging_steps"]
    eval_steps = config["eval_steps"]
    save_steps = config["save_steps"]
    learning_rate = config["learning_rate"]
    warmup_steps = config["warmup_steps"]
    num_train_epochs = config["num_train_epochs"]

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--source-field", type=str, default="sentences")
    parser.add_argument("--target-field", type=str, default="oracle")
    args = parser.parse_args()
    train(**vars(args))
