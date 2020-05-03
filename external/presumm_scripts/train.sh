PRESUM_PATH="/media/yallen/My Passport/Projects/PreSumm/src"
BERT_DATA_PATH="/media/yallen/My Passport/Projects/summarus/models/gazeta_presumm/data"
LOGS_PATH="/media/yallen/My Passport/Projects/summarus/models/gazeta_presumm/logs"
MODEL_PATH="/media/yallen/My Passport/Projects/summarus/models/gazeta_presumm"
export BERT_PATH="/media/yallen/My Passport/Models/rubert_cased_L-12_H-768_A-12_v2"

python3.6 "$PRESUM_PATH/train.py"  -task abs -mode train -bert_data_path "$BERT_DATA_PATH" \
  -dec_dropout 0.2 -model_path "$MODEL_PATH" -sep_optim true -lr_bert 1.0 -lr_dec 1.0 \
  -dec_layers 3 -save_checkpoint_steps 1000 -batch_size 2 -train_steps 50000 -report_every 50 \
  -accum_count 50 -use_bert_emb true -use_interval true -warmup_steps_bert 10000 \
  -warmup_steps_dec 5000 -max_pos 512 -visible_gpus 0 -log_file "$LOGS_PATH/log.txt"

