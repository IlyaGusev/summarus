BART_PATH="$1"
DATA_BIN_PATH="$2"
FAIRSEQ_PATH="$3"

TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=256
UPDATE_FREQ=1

CUDA_VISIBLE_DEVICES=0 python3 "${FAIRSEQ_PATH}/train.py" "${DATA_BIN_PATH}" \
    --encoder-normalize-before --decoder-normalize-before \
    --arch mbart_large \
    --task translation_from_pretrained_bart \
    --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN \
    --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 5 \
    --restore-file "${BART_PATH}/model.pt" \
    --max-tokens $MAX_TOKENS \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --memory-efficient-fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
