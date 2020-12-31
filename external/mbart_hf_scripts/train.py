import argparse
import json
import torch
from torch.utils.data import Dataset
from transformers import MBartTokenizer, MBartForConditionalGeneration, TrainingArguments, Trainer


class MBartSummarizationDataset(Dataset):
    def __init__(
        self,
        input_file,
        tokenizer,
        max_source_tokens_count=512,
        max_target_tokens_count=128
    ):
        self.pairs = []
        with open(input_file, "r") as f:
            for line in f:
                record = json.loads(line)
                source = record["text"]
                target = record["title"]
                self.pairs.append((source, target))
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, target = self.pairs[index]
        batch = self.tokenizer.prepare_seq2seq_batch(
            source,
            src_lang="ru_RU",
            tgt_lang="ru_RU",
            tgt_texts=target,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_source_tokens_count,
            max_target_length=self.max_target_tokens_count)
        return {
            "input_ids": batch["input_ids"][0],
            "attention_mask": batch["attention_mask"][0],
            "labels": batch["labels"][0]
        }


def train(
    model_name,
    train_file,
    val_file,
    batch_size,
    output_dir,
    learning_rate,
    logging_steps,
    eval_steps,
    save_steps,
    warmup_steps,
    num_train_epochs,
    max_source_tokens_count,
    max_target_tokens_count
):
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    train_dataset = MBartSummarizationDataset(
        train_file,
        tokenizer,
        max_source_tokens_count,
        max_target_tokens_count)
    val_dataset = MBartSummarizationDataset(
        val_file,
        tokenizer,
        max_source_tokens_count,
        max_target_tokens_count)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        do_train=True,
        do_eval=True,
        overwrite_output_dir=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="facebook/mbart-large-cc25")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=0.00003)
    parser.add_argument("--eval-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--num-train-epochs", type=int, default=2)
    parser.add_argument("--max-source-tokens-count", type=int, default=512)
    parser.add_argument("--max-target-tokens-count", type=int, default=128)
    args = parser.parse_args()
    train(**vars(args))
