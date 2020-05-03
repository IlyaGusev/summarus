PRESUM_PATH="/media/yallen/My Passport/Projects/PreSumm/src"
BERT_DATA_PATH="/media/yallen/My Passport/Projects/summarus/models/gazeta_presumm/data"
LOGS_PATH="/media/yallen/My Passport/Projects/summarus/models/gazeta_presumm/logs"
MODEL_PATH="/media/yallen/My Passport/Projects/summarus/models/gazeta_presumm"
RESULT_PATH="/media/yallen/My Passport/Projects/summarus/models/gazeta_presumm/predicted"
export BERT_PATH="/media/yallen/My Passport/Models/rubert_cased_L-12_H-768_A-12_v2"

python3.6 "$PRESUM_PATH/train.py" -task abs -mode validate -batch_size 32 -test_batch_size 32 \
  -bert_data_path "$BERT_DATA_PATH" -log_file "$LOGS_PATH/val_log.txt" -model_path "$MODEL_PATH" \
  -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -min_length 20 -max_length 100 \
  -alpha 0.9 -result_path "$RESULT_PATH" -report_every 32

