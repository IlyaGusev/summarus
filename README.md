# summarus

[![Tests Status](https://github.com/IlyaGusev/summarus/actions/workflows/python-package.yml/badge.svg)](https://github.com/IlyaGusev/summarus/actions/workflows/python-package.yml)
[![Code Climate](https://codeclimate.com/github/IlyaGusev/summarus/badges/gpa.svg)](https://codeclimate.com/github/IlyaGusev/summarus)

Abstractive and extractive summarization models, mostly for Russian language. Building on top of [AllenNLP](https://allennlp.org/)

You can also checkout the MBART-based Russian summarization model on Huggingface: [mbart_ru_sum_gazeta](https://huggingface.co/IlyaGusev/mbart_ru_sum_gazeta)

Based on the following papers:
* [SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents](https://arxiv.org/abs/1611.04230)
* [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
* [Self-Attentive Model for Headline Generation](https://arxiv.org/abs/1901.07786)
* [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)
* [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210)

## Contacts

* Telegram: [@YallenGusev](https://t.me/YallenGusev)

## Prerequisites
```
pip install -r requirements.txt
```

## Commands

#### train.sh

Script for training a model based on AllenNLP 'train' command.

| Argument | Required | Description                                      |
|:---------|:---------|--------------------------------------------------|
| -c       | true     | path to file with configuration                  |
| -s       | true     | path to directory where model will be saved      |
| -t       | true     | path to train dataset                            |
| -v       | true     | path to val dataset                              |
| -r       | false    | recover from checkpoint                          |

#### predict.sh

Script for model evaluation. The test dataset should have the same format as the train dataset.

| Argument | Required | Default | Description                                                      |
|:---------|:---------|:--------|:-----------------------------------------------------------------|
| -t       | true     |         | path to test dataset                                             |
| -m       | true     |         | path to tar.gz archive with model                                |
| -p       | true     |         | name of Predictor                                                |
| -c       | false    | 0       | CUDA device                                                      |
| -L       | true     |         | Language ("ru" or "en")                                          |
| -b       | false    | 32      | size of a batch with test examples to run simultaneously         |
| -M       | false    |         | path to meteor.jar for Meteor metric                             |
| -T       | false    |         | tokenize gold and predicted summaries before metrics calculation |
| -D       | false    |         | save temporary files with gold and predicted summaries           |

#### summarus.util.train_subword_model

Script for subword model training.

| Argument          | Default | Description                                                        |
|:------------------|:--------|:-------------------------------------------------------------------|
| --train-path      |         | path to train dataset                                              |
| --model-path      |         | path to directory where generated subword model will be saved      |
| --model-type      | bpe     | type of subword model, see sentencepiece                           |
| --vocab-size      | 50000   | size of the resulting subword model vocabulary                     |
| --config-path     |         | path to file with configuration for DatasetReader (with parse_set) |


## Headline generation

* First paper: [Importance of Copying Mechanism for News Headline Generation](http://www.dialog-21.ru/media/4599/gusevio-152.pdf)
* Slides: [Importance of Copying Mechanism for News Headline Generation](https://www.dropbox.com/s/agtvl3umlc6vci5/ICMNHG-Presentation.pdf)
* Second paper: [Advances of Transformer-Based Models for News Headline Generation](https://arxiv.org/abs/2007.05044)

#### Dataset splits:
* RIA original dataset: https://github.com/RossiyaSegodnya/ria_news_dataset
* RIA train/val/test: https://www.dropbox.com/s/rermx1r8lx9u7nl/ria.tar.gz
* RIA dataset preprocessed for mBART: https://www.dropbox.com/s/iq2ih8sztygvz0m/ria_data_mbart_512_200.tar.gz
* Lenta original dataset: https://github.com/yutkin/Lenta.Ru-News-Dataset
* Lenta train/val/test: https://www.dropbox.com/s/v9i2nh12a4deuqj/lenta.tar.gz
* Lenta dataset preprocessed for mBART: https://www.dropbox.com/s/4oo8jazmw3izqvr/lenta_mbart_data_512_200.tar.gz
* Telegram train dataset with split: https://www.dropbox.com/s/ykqk49a8avlmnaf/ru_all_split.tar.gz
* Telegram test dataset with multiple references: https://github.com/dialogue-evaluation/Russian-News-Clustering-and-Headline-Generation/blob/main/data/headline_generation/headline_generation_answers.jsonl

#### Models:
* [ria_copynet_10kk](https://www.dropbox.com/s/78ni5gnbcjz59ss/ria_copynet_10kk.tar.gz)
* [ria_pgn_24kk](https://www.dropbox.com/s/6wa1a2qzvqx5tti/ria_pgn_24kk.tar.gz)
* [ria_mbart](https://www.dropbox.com/s/bhrfd5o5etz8hso/ria_mbart_checkpoint_4.tar.gz)
* [rubert_telegram_headlines](https://huggingface.co/IlyaGusev/rubert_telegram_headlines)

Prediction script:
```
./predict.sh -t <path_to_test_dataset> -m ria_pgn_24kk.tar.gz -p subwords_summary -L ru 
```

#### Results
##### Train dataset: RIA, test dataset: RIA

| Model                     | R-1-f | R-2-f | R-L-f | BLEU  |
|:--------------------------|:------|:------|:------|:------|
| ria_copynet_10kk          | 40.0  | 23.3  | 37.5  | -     |
| ria_pgn_24kk              | 42.3  | 25.1  | 39.6  | -     |
| ria_mbart                 | 42.8  | 25.5  | 39.9  | -     |
| First Sentence            | 24.1  | 10.6  | 16.7  | -     |

#### Train dataset: RIA, eval dataset: Lenta

| Model                     | R-1-f | R-2-f | R-L-f | BLEU  |
|:--------------------------|:------|:------|:------|:------|
| ria_copynet_10kk          | 25.6  | 12.3  | 23.0  | -     |
| ria_pgn_24kk              | 26.4  | 12.3  | 24.0  | -     |
| ria_mbart                 | 30.3  | 14.5  | 27.1  | -     |
| First Sentence            | 25.5  | 11.2  | 19.2  | -     |

## Summarization - CNN/DailyMail

#### Dataset splits:
* CNN/DailyMail jsonl dataset: https://www.dropbox.com/s/35ezpg78rtukkgh/cnn_dm_jsonl.tar.gz

#### Models:
* [cnndm_pgn_25kk](https://www.dropbox.com/s/kctjduh84gam2pl/cnndm_pgn_25kk.tar.gz)

Prediction script:
```
./predict.sh -t <path_to_test_dataset> -m cnndm_pgn_25kk.tar.gz -p words_summary -L en -R
```

#### Results:

| Model                     | R-1-f | R-2-f | R-L-f | METEOR | BLEU |
|:--------------------------|:------|:------|:------|:-------|:-----|
| cnndm_pgn_25kk            | 38.5  | 16.5  | 33.4  | 17.6   | -    |


## Summarization - Gazeta, russian news dataset
* Paper: [Dataset for Automatic Summarization of Russian News](https://arxiv.org/abs/2006.11063)
* Gazeta dataset: https://github.com/IlyaGusev/gazeta
* Usage examples: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1B26oDFEKSNCcI0BPkGXgxi13pbadriyN)

#### Models:
* [gazeta_pgn_7kk](https://www.dropbox.com/s/aold2691f5amad8/gazeta_pgn_7kk.tar.gz)
* [gazeta_pgn_7kk_cov.tar.gz](https://www.dropbox.com/s/2yk25xaizevtqw3/gazeta_pgn_7kk_cov.tar.gz)
* [gazeta_pgn_25kk](https://www.dropbox.com/s/jmg7vk4ed9ph2ov/gazeta_pgn_25kk.tar.gz)
* [gazeta_pgn_words_13kk.tar.gz](https://www.dropbox.com/s/acexr5xecc8xizx/gazeta_pgn_words_13kk.tar.gz)
* [gazeta_summarunner_3kk](https://www.dropbox.com/s/mlo7ioxodqib1xl/gazeta_summarunner_3kk.tar.gz)

Prediction scripts:
```
./predict.sh -t <path_to_test_dataset> -m gazeta_pgn_7kk.tar.gz -p subwords_summary -L ru -T
./predict.sh -t <path_to_test_dataset> -m gazeta_summarunner_3kk.tar.gz -p subwords_summary_sentences -L ru -T
```

#### External models:
* [gazeta_mbart (fairseq)](https://www.dropbox.com/s/b2auu9dhrm2wj0p/gazeta_mbart_checkpoint_600_160.tar.gz)
* [gazeta_mbart (transformers)](https://huggingface.co/IlyaGusev/mbart_ru_sum_gazeta)
* [gazeta_mbart_lowercase (fairseq)](https://www.dropbox.com/s/k3gsgokq69468jw/gazeta_mbart_lower.tar.gz)

#### Results:

| Model                     | R-1-f | R-2-f | R-L-f | METEOR | BLEU |
|:--------------------------|:------|:------|:------|:-------|:-----|
| gazeta_pgn_7kk            | 29.4  | 12.7  | 24.6  | 21.2   | 9.0  |
| gazeta_pgn_7kk_cov        | 29.8  | 12.8  | 25.4  | 22.1   | 10.1 |
| gazeta_pgn_25kk           | 29.6  | 12.8  | 24.6  | 21.5   | 9.3  |
| gazeta_pgn_words_13kk     | 29.4  | 12.6  | 24.4  | 20.9   | 8.9  |
| gazeta_summarunner_3kk    | 31.6  | 13.7  | 27.1  | 26.0   | 11.5 |
| gazeta_mbart              | 32.6  | 14.6  | 28.2  | 25.7   | 12.4 |
| gazeta_mbart_lower        | 32.7  | 14.7  | 28.3  | 25.8   | 12.5 |


## Demo
```
python demo/server.py --include-package summarus --model-dir <model_dir> --host <host> --port <port>
```

## Citations
Headline generation (PGN):
```bibtex
@article{Gusev2019headlines,
    author={Gusev, I.O.},
    title={Importance of copying mechanism for news headline generation},
    journal={Komp'juternaja Lingvistika i Intellektual'nye Tehnologii},
    year={2019},
    volume={2019-May},
    number={18},
    pages={229--236}
}
```

Headline generation (transformers):
```bibtex
@InProceedings{Bukhtiyarov2020headlines,
    author={Bukhtiyarov, Alexey and Gusev, Ilya},
    title="Advances of Transformer-Based Models for News Headline Generation",
    booktitle="Artificial Intelligence and Natural Language",
    year="2020",
    publisher="Springer International Publishing",
    address="Cham",
    pages={54--61},
    isbn="978-3-030-59082-6",
    doi={10.1007/978-3-030-59082-6_4}
}
```

Summarization:
```bibtex
@InProceedings{Gusev2020gazeta,
    author="Gusev, Ilya",
    title="Dataset for Automatic Summarization of Russian News",
    booktitle="Artificial Intelligence and Natural Language",
    year="2020",
    publisher="Springer International Publishing",
    address="Cham",
    pages="{122--134}",
    isbn="978-3-030-59082-6",
    doi={10.1007/978-3-030-59082-6_9}
}
```
