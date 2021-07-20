import argparse
from transformers import MBartTokenizer, MBartForConditionalGeneration, TrainingArguments, Trainer

from dataset import MBartSummarizationDataset


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
    gradient_accumulation_steps,
    max_grad_norm,
    weight_decay,
    max_source_tokens_count,
    max_target_tokens_count,
    label_smoothing_factor,
    src_lang,
    tgt_lang,
    fp16_opt_level,
    fp16=False,
    freeze_body=False
):
    if fp16:
        print("Using FP16")
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    train_dataset = MBartSummarizationDataset(
        train_file,
        tokenizer,
        max_source_tokens_count,
        max_target_tokens_count,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )
    val_dataset = MBartSummarizationDataset(
        val_file,
        tokenizer,
        max_source_tokens_count,
        max_target_tokens_count,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    if freeze_body:
        for param in model.model.parameters():
            param.requires_grad = False
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
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        fp16=fp16,
        fp16_opt_level=fp16_opt_level,
        label_smoothing_factor=label_smoothing_factor,
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--logging-steps", type=int, default=400)
    parser.add_argument("--save-steps", type=int, default=6400)
    parser.add_argument("--learning-rate", type=float, default=0.00003)
    parser.add_argument("--eval-steps", type=int, default=3200)
    parser.add_argument("--warmup-steps", type=int, default=125)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--max-source-tokens-count", type=int, default=600)
    parser.add_argument("--max-target-tokens-count", type=int, default=160)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--src-lang", type=str, default="ru_RU")
    parser.add_argument("--tgt-lang", type=str, default="ru_RU")
    parser.add_argument("--label-smoothing-factor", type=float, default=0.1)
    parser.add_argument("--freeze-body", action='store_true')
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--fp16-opt-level", type=str, default="O1")

    args = parser.parse_args()
    train(**vars(args))
