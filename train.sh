#!/bin/bash
set -e

export CONFIG_PATH="$1"
export SERIALIZATION_PATH="$2"
export TRAIN_DATA_PATH="$3"
export VAL_DATA_PATH="$4"
export BPE_MODEL_PATH="$5"

allennlp train "${CONFIG_PATH}" -s "${SERIALIZATION_PATH}" --include-package summarus
