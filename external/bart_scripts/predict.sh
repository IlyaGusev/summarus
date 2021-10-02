CHECKPOINT_PATH="$1"
DATA_BIN_PATH="$2"
SENTENCE_BPE_MODEL="$3"
OUTPUT_FILE="$4"

CUDA_VISIBLE_DEVICES=1 fairseq-generate "${DATA_BIN_PATH}" \
  --path "${CHECKPOINT_PATH}" \
  --task translation_from_pretrained_bart \
  --gen-subset test -t ru_RU -s en_XX \
  --batch-size 3 \
  --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN \
  --bpe "sentencepiece" \
  --sentencepiece-model "${SENTENCE_BPE_MODEL}" > "${OUTPUT_FILE}";
