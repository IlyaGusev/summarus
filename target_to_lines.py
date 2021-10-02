import argparse

from summarus.util.io import read_jsonl


def target_to_lines(input_file, output_file, lowercase=True):
    with open(output_file, "w") as w:
        for r in read_jsonl(input_file):
            target = r["summary"]
            target = target.strip()
            target = target.lower() if lowercase else target
            w.write(target.replace("\n", " ") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()
    target_to_lines(**vars(args))
