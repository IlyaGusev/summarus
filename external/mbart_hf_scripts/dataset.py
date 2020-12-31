from torch.utils.data import Dataset


class MBartSummarizationDataset(Dataset):
    def __init__(
        self,
        input_file,
        tokenizer,
        max_source_tokens_count,
        max_target_tokens_count
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
