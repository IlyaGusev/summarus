#!/bin/bash
TEST_PATH=$1
mkdir -p logs;
python3.6 evaluate.py --model-path models/ria_5kk_subwords_seq2seq/ --test-path "$TEST_PATH" --report-every 1000 --batch-size 64 --metric all > logs/ria_10kk_subwords_copynet.log;
python3.6 evaluate.py --model-path models/ria_10kk_subwords_copynet/ --test-path "$TEST_PATH" --report-every 1000 --batch-size 64 --metric all > logs/ria_10kk_subwords_copynet.log;
python3.6 evaluate.py --model-path models/ria_10kk_words_copynet/ --test-path "$TEST_PATH" --report-every 1000 --batch-size 64 --metric all > logs/ria_10kk_words_copynet.log;
python3.6 evaluate.py --model-path models/ria_25kk_subwords_seq2seq/ --test-path "$TEST_PATH" --report-every 1000 --batch-size 64 --metric all > logs/ria_25kk_subwords_seq2seq.log;
python3.6 evaluate.py --model-path models/ria_25kk_words_seq2seq/ --test-path "$TEST_PATH" --report-every 1000 --batch-size 64 --metric all > logs/ria_25kk_words_seq2seq.log;
python3.6 evaluate.py --model-path models/ria_43kk_subwords_copynet_short_context/ --test-path "$TEST_PATH" --report-every 1000 --batch-size 64 --metric all > logs/ria_43kk_subwords_copynet_short_context.log
