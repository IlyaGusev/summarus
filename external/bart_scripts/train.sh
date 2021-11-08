MBART_PATH="$1"
DATA_BIN_PATH="$2"

MAX_UPDATE=80000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=605
UPDATE_FREQ=2
LANGS=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

CUDA_VISIBLE_DEVICES=1 fairseq-train "$DATA_BIN_PATH" \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task translation_from_pretrained_bart \
  --source-lang en_XX --target-lang ru_RU \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr "$LR" --warmup-updates "$WARMUP_UPDATES" \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --clip-norm 0.1 \
  --total-num-update "$MAX_UPDATE" \
  --max-tokens "$MAX_TOKENS" \
  --required-batch-size-multiple 1 \
  --update-freq "$UPDATE_FREQ" \
  --save-interval 1 --save-interval-updates 5000 \
  --keep-interval-updates 3 --no-epoch-checkpoints \
  --seed 42 --log-format simple --log-interval 100 \
  --restore-file "${MBART_PATH}/model.pt" \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --langs $LANGS \
  --find-unused-parameters \
  --memory-efficient-fp16 \
  --skip-invalid-size-inputs-valid-test \
  --find-unused-parameters \
  --no-save-optimizer-state
