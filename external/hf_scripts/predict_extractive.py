import argparse
import razdel

import torch
from tqdm import tqdm
from transformers import logging
from transformers import AutoTokenizer, AutoConfig, BertForTokenClassification

from extractive_model import ModelForSentencesClassification, ModelForSentencesClassificationConfig
from dataset import SummaryExtractiveDataset
from util import read_jsonl, set_random_seed, gen_batch


def predict(
    nrows,
    model_name,
    input_file,
    output_file,
    batch_size,
    max_source_tokens_count,
    max_target_sentences_count,
    threshold,
    seed,
):
    assert max_target_sentences_count or threshold is not None, \
        "Provide --max-target-sentences-count or/and --threshold"

    set_random_seed(seed)
    logging.set_verbosity_info()

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, strip_accents=False)
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type == "model-for-sentences-classification":
        model = ModelForSentencesClassification.from_pretrained(model_name)
    else:
        model = BertForTokenClassification.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    records = list(read_jsonl(input_file))
    if nrows:
        records = records[:nrows]

    summaries = []
    for records_batch in tqdm(gen_batch(records, batch_size)):
        batch = []
        for r in records_batch:
            sentences = [s.text for s in razdel.sentenize(r["text"])]
            if hasattr(model.config, "max_sentences_count"):
                sentences = sentences[:model.config.max_sentences_count]
            batch.append(sentences)

        sep_token = tokenizer.sep_token
        sep_token_id = tokenizer.sep_token_id
        texts = [sep_token.join(sentences) for sentences in batch]
        inputs = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_source_tokens_count,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        sep_mask = inputs["input_ids"] == sep_token_id

        # Fix token_type_ids
        for seq_num, seq in enumerate(inputs["input_ids"]):
            current_token_type_id = 0
            for pos, input_id in enumerate(seq):
                inputs["token_type_ids"][seq_num][pos] = current_token_type_id
                if input_id == sep_token_id:
                    current_token_type_id = 1 - current_token_type_id

        # Infer model
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        include_logits = outputs.logits[:, :, 1]

        # Choose sentences
        for sample_logits, sentences, sample_sep_mask in zip(include_logits, batch, sep_mask):
            if config.model_type == "model-for-sentences-classification":
                sentences_count = torch.sum(sample_sep_mask).item()
                sentences_count = min(sentences_count, config.max_sentences_count)
                sentences_logits = sample_logits[:sentences_count]
            else:
                sentences_logits = sample_logits[sample_sep_mask]

            logits, indices = sentences_logits.sort(descending=True)
            logits, indices = logits.cpu().tolist(), indices.cpu().tolist()
            pairs = list(zip(logits, indices))
            if threshold:
                pairs = [(logit, idx) for logit, idx in pairs if logit >= threshold]
            if max_target_sentences_count:
                pairs = pairs[:max_target_sentences_count]
            indices = list(sorted([idx for _, idx in pairs]))
            summary = " ".join([sentences[idx] for idx in indices])
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
    parser.add_argument("--max-target-sentences-count", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    predict(**vars(args))
