import argparse
import razdel

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, logging

from extractive_model import ModelForSentencesClassification
from dataset import SummaryExtractiveDataset
from util import read_jsonl, set_random_seed, gen_batch


def predict(
    nrows,
    model_name,
    input_file,
    output_file,
    batch_size,
    max_source_tokens_count,
    max_source_sentences_count,
    seed,
):
    set_random_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, strip_accents=False)
    model = ModelForSentencesClassification.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    records = list(read_jsonl(input_file))
    if nrows:
        records = records[:nrows]

    summaries = []
    for batch in tqdm(gen_batch(records, batch_size)):
        sentences_batch = [[s.text for s in razdel.sentenize(r["text"])] for r in batch]
        texts = [tokenizer.sep_token.join(sentences) for sentences in sentences_batch]
        inputs = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_source_tokens_count,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        sep_token_id = tokenizer.sep_token_id

        # Fix token_type_ids
        for seq_num, seq in enumerate(inputs["input_ids"]):
            current_token_type_id = 0
            for pos, input_id in enumerate(seq):
                inputs["token_type_ids"][seq_num][pos] = current_token_type_id
                if input_id == sep_token_id:
                    current_token_type_id = 1 - current_token_type_id

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = outputs.logits[:, :, 1]
        for sample_logits, sentences in zip(logits, sentences_batch):
            sample_logits = sample_logits[:len(sentences)]
            #indices = [i for i, logit in enumerate(sample_logits) if logit >= 0.0]
            #if not indices:
            indices = sorted(torch.topk(sample_logits, 3).indices.cpu().numpy().tolist())
            summary = " ".join([sentences[idx] for idx in indices])
            print(summary)
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-tokens-count", type=int, default=500)
    parser.add_argument("--max-source-sentences-count", type=int, default=40)
    args = parser.parse_args()
    predict(**vars(args))
