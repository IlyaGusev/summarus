CHECKPOINT_PATH="$1"
DATA_BIN_PATH="$2"
FAIRSEQ_PATH="$3"
SENTENCE_BPE_MODEL="$4"
OUTPUT_FILE="$5"
MAX_SENTENCES="$6"

python3 "${FAIRSEQ_PATH}/generate.py" "${DATA_BIN_PATH}" \
  --path "${CHECKPOINT_PATH}" \
  --task translation_from_pretrained_bart \
  --gen-subset test -t target -s source \
  --max-sentences "${MAX_SENTENCES}" \
  --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN \
  --bpe "sentencepiece" \
  --sentencepiece-vocab "${SENTENCE_BPE_MODEL}" \
  --sacrebleu > "${OUTPUT_FILE}";

