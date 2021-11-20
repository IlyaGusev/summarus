import argparse
import json
import itertools

import torch
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, EncoderDecoderModel
from tqdm import tqdm


def read_jsonl(file_path):
    with open(file_path) as r:
        for line in r:
            yield json.loads(line)


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


def predict(
    nrows,
    model_name,
    model_type,
    input_file,
    output_file,
    batch_size,
    max_source_tokens_count,
    max_target_tokens_count
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, strip_accents=False)
    if model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    elif model_type == "encoder_decoder":
        model = EncoderDecoderModel.from_pretrained(model_name)
    else:
        assert False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    records = list(read_jsonl(input_file))
    if nrows:
        records = records[:nrows]

    summaries = []
    for batch in tqdm(gen_batch(records, batch_size)):
        texts = [r["text"] for r in batch]
        input_ids = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_source_tokens_count,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_target_tokens_count,
            no_repeat_ngram_size=3,
            num_beams=5,
            early_stopping=True
        )

        for ids in output_ids:
            summary = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            summaries.append(summary)

    with open(output_file, "w") as w:
        for s in summaries:
            w.write(s.strip() + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-tokens-count", type=int, default=400)
    parser.add_argument("--max-target-tokens-count", type=int, default=200)
    args = parser.parse_args()
    predict(**vars(args))
