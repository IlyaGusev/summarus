import argparse
import torch
from transformers import MBartTokenizer, MBartForConditionalGeneration

from dataset import MBartSummarizationDataset


def predict(
    model_name,
    test_file,
    output_file,
    batch_size,
    max_source_tokens_count,
    max_target_tokens_count,
    use_cuda
):
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    test_dataset = MBartSummarizationDataset(
        test_file,
        tokenizer,
        max_source_tokens_count,
        max_target_tokens_count
    )
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    predictions = []
    for batch in test_dataset:
        summaries = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            num_beams=5,
            length_penalty=1.0,
            max_length=max_target_tokens_count + 2,
            min_length=5,
            no_repeat_ngram_size=0,
            early_stopping=True
        )
        for s in summaries:
            p = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            predictions.append(p)
    with open(output_file, "w") as w:
        for p in predictions:
            w.write(p.strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--max-source-tokens-count", type=int, default=512)
    parser.add_argument("--max-target-tokens-count", type=int, default=128)
    parser.add_argument("--use-cuda", action='store_true')
    args = parser.parse_args()
    predict(**vars(args))
