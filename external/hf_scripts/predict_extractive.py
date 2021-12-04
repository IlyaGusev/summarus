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
    seed,
):
    set_random_seed(seed)
    logging.set_verbosity_info()
    AutoConfig.register("model-for-sentences-classification", ModelForSentencesClassificationConfig)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, strip_accents=False)
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type == "model-for-sentences-classification":
        model = ModelForSentencesClassification.from_pretrained(model_name)
    else:
        model = BertForTokenClassification.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    records = list(read_jsonl(input_file))
    if nrows:
        records = records[:nrows]

    summaries = []
    for batch in tqdm(gen_batch(records, batch_size)):
        sentences_batch = []
        for r in batch:
            sentences = [s.text for s in razdel.sentenize(r["text"])]
            sentences = sentences[:model.config.max_sentences_count]
            sentences_batch.append(sentences)
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
        sentence_token_mask = inputs["input_ids"] == sep_token_id

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
        for sample_logits, sentences, sample_sentence_token_mask in zip(logits, sentences_batch, sentence_token_mask):
            if config.model_type == "model-for-sentences-classification":
                sentences_count = min(torch.sum(sample_sentence_token_mask).item(), config.max_sentences_count)
                sentences_logits = sample_logits[:sentences_count]
            else:
                sentences_logits = sample_logits[sample_sentence_token_mask]
            indices = sorted(torch.topk(sentences_logits, 3).indices.cpu().numpy().tolist())
            print(indices)
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
    parser.add_argument("--max-source-tokens-count", type=int, default=510)
    args = parser.parse_args()
    predict(**vars(args))
