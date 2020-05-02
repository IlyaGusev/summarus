import argparse

from allennlp.common.params import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from summarus.readers import *


def target_to_lines(reader_config_file, input_file, output_file, lowercase=True):
    params = Params.from_file(reader_config_file)
    reader = DatasetReader.from_params(params.pop("dataset_reader"))
    with open(output_file, "w") as w:
        for t in reader.parse_set(input_file):
            target = t[1]
            target = target.strip()
            target = target.lower() if lowercase else target
            w.write(target.replace("\n", " ") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader-config-file', type=str, required=True)
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()
    target_to_lines(**vars(args))
