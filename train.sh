#!/bin/bash
set -e

usage() {
  echo "Usage: $0 -c CONFIG_PATH -s SERIALIZATION_PATH -t TRAIN_DATA_PATH -v VAL_DATA_PATH" 1>&2
}

exit_abnormal() {
  usage
  exit 1
}

c_flag=false;
s_flag=false;
t_flag=false;
v_flag=false;

while getopts ":c:s:t:v:r" opt; do
  case $opt in
    # Options for AllenNLP 'train'
    # Path to training config
    c) CONFIG_PATH="$OPTARG"; c_flag=true
    ;;
    # Path to a new directory for the model to be saved
    s) SERIALIZATION_PATH="$OPTARG"; s_flag=true
    ;;
    # --recover option
    r) r_flag=true
    ;;

    # Options below are required by .jsonnet config files
    # Path to training data
    t) export TRAIN_DATA_PATH="$OPTARG"; t_flag=true
    ;;
    # Path to validation data (for early stopping)
    v) export VAL_DATA_PATH="$OPTARG"; v_flag=true
    ;;

    \?) echo "Invalid option -$OPTARG" >&2; exit_abnormal
    ;;
    :) echo "Missing option argument for -$OPTARG" >&2; exit_abnormal
    ;;
  esac
done

if ! $c_flag
then
    echo "Missing -c option"; exit_abnormal;
fi

if ! $s_flag
then
    echo "Missing -s option"; exit_abnormal;
fi

if ! $t_flag
then
    echo "Missing -t option"; exit_abnormal;
fi

if ! $v_flag
then
    echo "Missing -v option"; exit_abnormal;
fi

allennlp train "${CONFIG_PATH}" -s "${SERIALIZATION_PATH}" --include-package summarus ${r_flag:+--recover}
