import argparse
import sys
import json
from collections import defaultdict

from scipy.stats import pearsonr


def main(input_path):
    values = defaultdict(list)
    with open(input_path) as r:
        for line in r:
            record = json.loads(line)
            stats = record["stats"]
            for key, value in stats.items():
                values[key].append(value)
    for key1, value1 in values.items():
        for key2, value2 in values.items():
            if key1 == key2:
                continue
            print(key1, key2, pearsonr(value1, value2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    args = parser.parse_args()
    main(**vars(args))
