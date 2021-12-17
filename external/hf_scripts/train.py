import argparse
import random
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments, logging
from transformers import EncoderDecoderModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from dataset import SummarySeq2SeqDataset, SummaryLMDataset
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

    model_type = config["model_type"]
    assert model_type in ("causal_lm", "encoder_decoder", "seq2seq_lm")
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, strip_accents=False)
    tokenizer = fix_tokenizer(tokenizer)

    # Data preparation
    train_records = list(read_jsonl(train_file))
    val_records = list(read_jsonl(val_file))
    random.shuffle(train_records)

    dataset_class = SummaryLMDataset if model_type in ("causal_lm",) else SummarySeq2SeqDataset
    max_source_tokens_count = config["max_source_tokens_count"]
    max_target_tokens_count = config["max_target_tokens_count"]
    only_summary_loss = config.get("only_summary_loss", False)
    train_dataset_args = {
        "original_records": train_records,
        "sample_rate": train_sample_rate,
        "tokenizer": tokenizer,
        "max_source_tokens_count": max_source_tokens_count,
        "max_target_tokens_count": max_target_tokens_count,
        "source_field": source_field,
        "target_field": target_field
    }
    val_dataset_args = {
        "original_records": val_records,
        "sample_rate": val_sample_rate,
        "tokenizer": tokenizer,
        "max_source_tokens_count": max_source_tokens_count,
        "max_target_tokens_count": max_target_tokens_count,
        "source_field": source_field,
        "target_field": target_field
    }
    if only_summary_loss:
        train_dataset_args["only_summary_loss"] = True
        val_dataset_args["only_summary_loss"] = True
    train_dataset = dataset_class(**train_dataset_args)
    val_dataset = dataset_class(**val_dataset_args)

    # Model loading
    if model_type == "encoder_decoder":
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
    elif model_type == "seq2seq_lm":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_name)
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

    # Default model generation params
    model.config.num_beams = 5
    model.config.max_length = max_target_tokens_count
    if model_type == "causal_lm":
        model.config.max_length = max_target_tokens_count + max_source_tokens_count

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
    parser.add_argument("--source-field", type=str, default="text")
    parser.add_argument("--target-field", type=str, default="summary")
    args = parser.parse_args()
    train(**vars(args))
