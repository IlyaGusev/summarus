export TRAIN_DATA_PATH="/data/train_shuf.txt"
export VAL_DATA_PATH="/data/val.txt"
export TEST_DATA_PATH="/data/test.txt"
export TASK="gazeta"
export BPE_MODEL_PATH="models/gazeta_full_5k/bpe.model"

allennlp train "$1" -s models/gazeta_pgn_model --include-package summarus
