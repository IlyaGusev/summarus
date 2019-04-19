# summarus

Summarization models

## Prerequisites
```
pip install -r requirements.txt
```

## Headline generation

### Dataset splits
* RIA original dataset: https://github.com/RossiyaSegodnya/ria_news_dataset
* RIA splits: https://www.dropbox.com/s/rermx1r8lx9u7nl/ria.tar.gz
* Lenta original dataset: https://github.com/yutkin/Lenta.Ru-News-Dataset
* Lenta splits: https://www.dropbox.com/s/v9i2nh12a4deuqj/lenta.tar.gz

### Models
* seq2seq-bpe-5m: https://www.dropbox.com/s/oajac54fb5dprzw/ria_5kk_subwords_seq2seq.tar.gz
* copynet-words-10m: https://www.dropbox.com/s/sed8yh0yq4a7bmt/ria_10kk_words_copynet.tar.gz
* copynet-bpe-10m: https://www.dropbox.com/s/v71akkarrtcjlxm/ria_10kk_subwords_copynet.tar.gz
* seq2seq-words-25m: https://www.dropbox.com/s/2powhpdjo8zmny8/ria_25kk_words_seq2seq.tar.gz
* seq2seq-bpe-25m: https://www.dropbox.com/s/qc8flgqxt59ukdh/ria_25kk_subwords_seq2seq.tar.gz
* copynet-words-25m: https://www.dropbox.com/s/52v50z3ne6qyuv5/ria_25kk_words_copynet.tar.gz
* copynet-bpe-43m: https://www.dropbox.com/s/w67dcqf1mlv66uy/ria_43kk_subwords_copynet_short_context.tar.gz

### Results

#### Train dataset: RIA, test dataset: RIA

| Model             | R-1-f | R-1-r | R-2-f | R-2-r | R-L-f | R-L-r | R-mean-f | BLEU  |
|:------------------|:------|:------|:------|:------|:------|:------|:---------|:------|
| seq2seq-words-25m | 36.96 | 35.19 | 19.68 | 19.02 | 34.30 | 33.60 | 30.31    | 44.69 |
| seq2seq-bpe-5m    | 38.78 | 36.91 | 21.87 | 20.90 | 35.96 | 35.24 | 32.20    | 49.77 |
| seq2seq-bpe-25m   | 40.30 | 38.83 | 22.94 | 22.18 | 37.50 | 37.01 | 33.58    | 51.66 |
| pgn-dot-words-5m  | 37.91 | 35.98 | 20.43 | 19.55 | 35.19 | 34.36 | 31.18    | 47.08 |
| copynet-words-10m | 39.48 | 38.39 | 22.57 | 22.05 | 36.95 | 36.69 | 33.00    | 51.99 |
| copynet-bpe-10m   | 40.03 | 38.68 | 23.25 | 22.50 | 37.44 | 37.04 | 33.57    | 52.57 |
| copynet-words-25m | 40.38 | 39.46 | 23.26 | 22.83 | 37.80 | 37.70 | 33.81    | 52.99 |
| copynet-bpe-43m   | 41.61 | 40.33 | 24.46 | 23.76 | 38.85 | 34.97 | 34.97    | 53.80 |
| First Sentence    | 24.08 | 45.58 | 10.57 | 21.30 | 16.70 | 41.67 | 17.12    | -     |

#### Train dataset: RIA, eval dataset: Lenta

| Model             | R-1-f | R-1-r | R-2-f | R-2-r | R-L-f | R-L-r | R-mean-f | BLEU  |
|:------------------|:------|:------|:------|:------|:------|:------|:---------|:------|
| seq2seq-words-25m | 18.29 | 17.11 | 7.21  | 6.96  | 16.23 | 16.13 | 13.91    | 23.35 |
| seq2seq-bpe-5m    | 19.38 | 17.35 | 8.27  | 7.43  | 16.94 | 16.55 | 14.86    | 25.14 |
| seq2seq-bpe-25m   | 20.75 | 19.06 | 8.77  | 8.11  | 18.15 | 17.97 | 15.89    | 28.21 |
| copynet-words-10m | 26.37 | 26.38 | 12.67 | 12.74 | 24.04 | 25.06 | 21.02    | 38.36 |
| copynet-bpe-10m   | 25.60 | 24.57 | 12.33 | 11.84 | 23.03 | 23.33 | 20.32    | 36.13 |
| copynet-words-25m | 28.24 | 27.51 | 13.67 | 13.51 | 25.67 | 25.91 | 22.53    | 40.13 |
| copynet-bpe-43m   | 28.27 | 27.61 | 13.95 | 13.63 | 25.77 | 26.19 | 22.66    | 40.44 |
| First Sentence    | 25.45 | 40.52 | 11.16 | 18.63 | 19.17 | 37.80 | 18.59    | 25.45 |

### Commands

#### preprocess.py

Script for generation of a vocabulary.
Uses a configuration file to determine the size of the vocabulary and options of dataset preprocessing.

| Argument          | Default | Description                                      |
|:------------------|:--------|:-------------------------------------------------|
| --train-path      |         | path to train dataset                            |
| --config-path     |         | path to file with configuration                  |
| --vocabulary-path |         | path to directory where vocabulary will be saved |

#### train_subword_model.py

Script for subword model training.

| Argument          | Default | Description                                                   |
|:------------------|:--------|:--------------------------------------------------------------|
| --train-path      |         | path to train dataset                                         |
| --model-path      |         | path to directory where generated subword model will be saved |
| --model-type      | bpe     | type of subword model, see sentencepiece                      |
| --vocab-size      | 50000   | size of the resulting subword model vocabulary                |

#### train.py

Script for model training. Model directory should exist as well as config file and vocabulary directory.
You should use preprocess.py to generate a vocabulary based on the train dataset.

| Argument          | Default | Description                          |
|:------------------|:--------|:-------------------------------------|
| --train-path      |         | path to train dataset                |
| --model-path      |         | path to directory with model's files |
| --val-path        | None    | path to val dataset                  |
| --seed            | 1048596 | random seed                          |
| --vocabulary-path | None    | custom path to vocabulary            |
| --config-path     | None    | custom path to config                |

#### evaluate.py

Script for model evaluation. The test dataset should have the same format as the train dataset.

| Argument          | Default | Description                                               |
|:------------------|:--------|:----------------------------------------------------------|
| --test-path       |         | path to test dataset                                      |
| --model-path      |         | path to directory with model's files                      |
| --metric          | all     | what metric to evaluate, choices=("rouge", "bleu", "all") |
| --max-count       | None    | how many test examples to consider                        |
| --report-every    | None    | print metrics every N'th step                             |
| --config-path     | None    | custom path to config                                     |
| --batch-size      | 32      | size of a batch with test examples to run simultaneously  |
