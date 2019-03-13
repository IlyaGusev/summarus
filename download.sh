#!/bin/sh
mkdir -p models

wget https://www.dropbox.com/s/oajac54fb5dprzw/ria_5kk_subwords_seq2seq.tar.gz
tar -xzvf ria_5kk_subwords_seq2seq.tar.gz --one-top-level=models/ria_5kk_subwords_seq2seq
rm ria_5kk_subwords_seq2seq.tar.gz

wget https://www.dropbox.com/s/v71akkarrtcjlxm/ria_10kk_subwords_copynet.tar.gz
tar -xzvf ria_10kk_subwords_copynet.tar.gz --one-top-level=models/ria_10kk_subwords_copynet
rm ria_10kk_subwords_copynet.tar.gz

wget https://www.dropbox.com/s/sed8yh0yq4a7bmt/ria_10kk_words_copynet.tar.gz
tar -xzvf ria_10kk_words_copynet.tar.gz --one-top-level=models/ria_10kk_words_copynet
rm ria_10kk_words_copynet.tar.gz

wget https://www.dropbox.com/s/qc8flgqxt59ukdh/ria_25kk_subwords_seq2seq.tar.gz
tar -xzvf ria_25kk_subwords_seq2seq.tar.gz --one-top-level=models/ria_25kk_subwords_seq2seq
rm ria_25kk_subwords_seq2seq.tar.gz

wget https://www.dropbox.com/s/2powhpdjo8zmny8/ria_25kk_words_seq2seq.tar.gz
tar -xzvf ria_25kk_words_seq2seq.tar.gz --one-top-level=models/ria_25kk_words_seq2seq
rm ria_25kk_words_seq2seq.tar.gz

wget https://www.dropbox.com/s/w67dcqf1mlv66uy/ria_43kk_subwords_copynet_short_context.tar.gz
tar -xzvf ria_43kk_subwords_copynet_short_context.tar.gz --one-top-level=models/ria_43kk_subwords_copynet_short_context
rm ria_43kk_subwords_copynet_short_context.tar.gz
