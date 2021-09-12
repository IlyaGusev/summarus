import json


def write_jsonl(records, path):
    with open(path, "w") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(file_name):
    records = []
    with open(file_name, "r") as r:
        for line in r:
            record = json.loads(line)
            records.append(record)
    return records
