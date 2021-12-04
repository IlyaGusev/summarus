import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from util import read_jsonl, gen_batch, set_random_seed


def predict(
    nrows,
    model_name,
    model_type,
    input_file,
    output_file,
    batch_size,
    max_source_tokens_count,
    seed,
    no_repeat_ngram_size,
    repetition_penalty,
    length_penalty,
    num_beams
):
    set_random_seed(seed)
    is_causal_lm = (model_type == "causal_lm")
    if is_causal_lm:
        assert batch_size == 1, "For causal LM only batch_size == 1 is supported!"

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, strip_accents=False)
    if model_type == "seq2seq_lm":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        assert False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    records = list(read_jsonl(input_file))
    if nrows:
        records = records[:nrows]

    summaries = []
    if is_causal_lm:
        for r in tqdm(records):
            text_tokens = tokenizer(
                r["text"],
                add_special_tokens=False,
                max_length=max_source_tokens_count,
                padding=False,
                truncation=True
            )["input_ids"]
            input_ids = text_tokens + [tokenizer.sep_token_id]
            input_ids = torch.LongTensor([input_ids]).to(device)
            output_ids = model.generate(
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_beams=num_beams
            )
            summary = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            summary = summary.split(tokenizer.sep_token)[1]
            summary = summary.split(tokenizer.eos_token)[0]
            summaries.append(summary)
    else:
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
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_beams=num_beams
            )

            for ids in output_ids:
                summary = tokenizer.decode(ids, skip_special_tokens=True)
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
    parser.add_argument("--model-type", type=str, required=True, choices=("causal_lm", "seq2seq_lm"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-tokens-count", type=int, default=600)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=5)
    args = parser.parse_args()
    predict(**vars(args))
