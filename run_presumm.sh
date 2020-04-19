PRESUM_PATH="/media/yallen/My Passport/Projects/PreSumm/src"
BERT_DATA_PATH="/media/yallen/My Passport/Projects/summarus/models/gazeta_presumm/data"
LOGS_PATH="/media/yallen/My Passport/Projects/summarus/models/gazeta_presumm/logs"
MODEL_PATH="/media/yallen/My Passport/Projects/summarus/models/gazeta_presumm"
RESULT_PATH="/media/yallen/My Passport/Projects/summarus/models/gazeta_presumm/predicted"

python3.6 "$PRESUM_PATH/train.py"  -task abs -mode train -bert_data_path "$BERT_DATA_PATH" \
  -dec_dropout 0.2 -model_path "$MODEL_PATH" -sep_optim true -lr_bert 1.0 -lr_dec 1.0 \
  -dec_layers 3 -save_checkpoint_steps 1000 -batch_size 2 -train_steps 50000 -report_every 50 \
  -accum_count 50 -use_bert_emb true -use_interval true -warmup_steps_bert 10000 \
  -warmup_steps_dec 5000 -max_pos 512 -visible_gpus 0 -log_file "$LOGS_PATH/log.txt" \
  -train_from "$MODEL_PATH/model_step_1000.pt"

#python3.6 "$PRESUM_PATH/train.py" -task abs -mode validate -batch_size 32 -test_batch_size 32 \
#-bert_data_path "$BERT_DATA_PATH" -log_file "$LOGS_PATH/val_log.txt" -model_path "$MODEL_PATH" \
#-sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -min_length 20 -max_length 100 \
#-alpha 0.9 -result_path "$RESULT_PATH" -report_every 32

