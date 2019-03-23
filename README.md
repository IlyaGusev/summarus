# summarus

Summarization models

# Install 
```
pip install -r requirements.txt
```

# Headline generation

## Dataset splits
* RIA original dataset: https://github.com/RossiyaSegodnya/ria_news_dataset
* RIA splits: https://www.dropbox.com/s/rermx1r8lx9u7nl/ria.tar.gz
* Lenta original dataset: https://github.com/yutkin/Lenta.Ru-News-Dataset
* Lenta splits: https://www.dropbox.com/s/v9i2nh12a4deuqj/lenta.tar.gz

## Models
* seq2seq-bpe-5m: https://www.dropbox.com/s/oajac54fb5dprzw/ria_5kk_subwords_seq2seq.tar.gz
* copynet-words-10m: https://www.dropbox.com/s/sed8yh0yq4a7bmt/ria_10kk_words_copynet.tar.gz
* copynet-bpe-10m: https://www.dropbox.com/s/v71akkarrtcjlxm/ria_10kk_subwords_copynet.tar.gz
* seq2seq-words-25m: https://www.dropbox.com/s/2powhpdjo8zmny8/ria_25kk_words_seq2seq.tar.gz
* seq2seq-bpe-25m: https://www.dropbox.com/s/qc8flgqxt59ukdh/ria_25kk_subwords_seq2seq.tar.gz
* copynet-words-25m: https://www.dropbox.com/s/52v50z3ne6qyuv5/ria_25kk_words_copynet.tar.gz
* copynet-bpe-43m: https://www.dropbox.com/s/w67dcqf1mlv66uy/ria_43kk_subwords_copynet_short_context.tar.gzm

## Results

| Model             | R-1-f | R-1-r | R-2-f | R-2-r | R-L-f | R-L-r | R-mean-f | BLEU  |
|:------------------|:------|:------|:------|:------|:------|:------|:---------|:------|
| seq2seq-bpe-5m    | 38.78 | 36.91 | 21.87 | 20.90 | 35.96 | 35.24 | 32.20    | 49.77 |
| copynet-words-10m | 39.48 | 38.39 | 22.57 | 22.05 | 36.95 | 36.69 | 33.00    | 51.99 |
| copynet-bpe-10m   | 40.03 | 38.68 | 23.25 | 22.50 | 37.44 | 37.04 | 33.57    | 52.57 |
| seq2seq-words-25m | 36.96 | 35.19 | 19.68 | 19.02 | 34.30 | 33.60 | 30.31    | 44.69 |
| seq2seq-bpe-25m   | 40.30 | 38.83 | 22.94 | 22.18 | 37.50 | 37.01 | 33.58    | 51.66 |
| copynet-words-25m | 40.38 | 39.46 | 23.26 | 22.83 | 37.80 | 37.70 | 33.81    | 52.99 |
| copynet-bpe-43m   | 41.61 | 40.33 | 24.46 | 23.76 | 38.85 | 34.97 | 34.97    | 53.80 |
