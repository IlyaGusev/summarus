#!/bin/bash
set -e

MODEL_ARCHIVE_PATH="$1"
TEST_FILE="$2"
PRED_FILE=$(mktemp)
REF_FILE=$(mktemp)
METEOR_JAR="/data/meteor-1.5/meteor-1.5.jar"
BATCH_SIZE=32
PREDICTOR="subwords_summary"
READER_CONFIG_FILE="models/gazeta_full_5k_mds200_shuf_attnfix_temp/config.json"

echo "Calling AllenNLP predict...";
allennlp predict \
  "${MODEL_ARCHIVE_PATH}" \
  "${TEST_FILE}" \
  --output-file "${PRED_FILE}" \
  --include-package summarus \
  --cuda-device 0 \
  --use-dataset-reader \
  --predictor "${PREDICTOR}" \
  --silent \
  --batch-size ${BATCH_SIZE};
echo "File with predictions: ${PRED_FILE}";

echo "Calling target_to_lines.py...";
python3.6 target_to_lines.py \
  --reader-config-file "${READER_CONFIG_FILE}" \
  --input-file "${TEST_FILE}" \
  --output-file "${REF_FILE}";
echo "File with gold summaries: ${REF_FILE}";

echo "Calling new_evaluate.py...";
python3.6 evaluate.py \
  --predicted-path "${PRED_FILE}" \
  --gold-path "${REF_FILE}" \
  --metric all \
  --tokenize-after \
  --meteor-jar ${METEOR_JAR};

echo "Removing temporary files...";
rm "${PRED_FILE}";
rm "${REF_FILE}";
