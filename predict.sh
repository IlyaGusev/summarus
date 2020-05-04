#!/bin/bash
set -e

usage() {
  echo "Usage: $0 -m MODEL_ARCHIVE_PATH -t TEST_FILE -p PREDICTOR -r READER_CONFIG_FILE [ -b BATCH_SIZE ] [ -M METEOR_JAR ]" 1>&2
}

exit_abnormal() {
  usage
  exit 1
}

m_flag=false;
t_flag=false;
p_flag=false;
b_flag=false;
r_flag=false;
D_flag=false;

while getopts ":m:t:p:b:r:M:TD" opt; do
  case $opt in
    # Options for AllenNLP 'predict'
    # Path to tar.gz archive with model
    m) MODEL_ARCHIVE_PATH="$OPTARG"; m_flag=true
    ;;
    # Path to file with data for testing
    t) TEST_FILE="$OPTARG"; t_flag=true
    ;;
    # Registered AllenNLP Predictor name
    p) PREDICTOR="$OPTARG"; p_flag=true
    ;;
    # Batch size (default: 32)
    b) BATCH_SIZE="$OPTARG"; b_flag=true
    ;;

    # Options for target_to_lines
    # Path to training data
    r) READER_CONFIG_FILE="$OPTARG"; r_flag=true
    ;;

    # Options for evaluate.py
    # Path to validation data (for early stopping)
    M) METEOR_JAR="$OPTARG"; M_flag=true
    ;;
    # --tokenize-after for evaluate.py
    T) T_flag=true
    ;;

    # Other options
    # Do not remove temporary files
    D) D_flag=true
    ;;

    \?) echo "Invalid option -$OPTARG" >&2; exit_abnormal
    ;;
    :) echo "Missing option argument for -$OPTARG" >&2; exit_abnormal
    ;;
  esac
done

if ! $m_flag
then
    echo "Missing -m option (path to model archive)"; exit_abnormal;
fi

if ! $t_flag
then
    echo "Missing -t option (path to test dataset)"; exit_abnormal;
fi

if ! $p_flag
then
    echo "Missing -p option (name of Predictor)"; exit_abnormal;
fi

if ! $r_flag
then
    echo "Missing -r option (path to reader config to read gold targets)"; exit_abnormal;
fi

if ! $b_flag
then
    BATCH_SIZE=32;
fi

PRED_FILE=$(mktemp)
REF_FILE=$(mktemp)

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

echo "Calling evaluate.py...";
python3.6 evaluate.py \
  --predicted-path "${PRED_FILE}" \
  --gold-path "${REF_FILE}" \
  --metric all \
  ${M_flag:+--meteor-jar $METEOR_JAR} \
  ${T_flag:+--tokenize-after};

if ! $D_flag
then
  echo "Removing temporary files...";
  rm "${PRED_FILE}";
  rm "${REF_FILE}";
else
  echo "File with predicted summaries: ${PRED_FILE}";
  echo "File with gold summaries: ${REF_FILE}";
fi
