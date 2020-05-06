local TRAIN_DATA_PATH = std.extVar("TRAIN_DATA_PATH");
local VAL_DATA_PATH = std.extVar("VAL_DATA_PATH");
local BPE_MODEL_PATH = "https://bpe-models.s3.eu-west-3.amazonaws.com/bpe_gazeta_5k.model";
local READER = "gazeta_sentences_tagger_reader";
local LOWERCASE = true;
local MAX_SENTENCES_COUNT = 50;
local SENTENCE_MAX_TOKENS = 100;
local VOCAB_SIZE = 5000;
local BATCH_SIZE = 32;
local EMBEDDING_DIM = 128;
local SENTENCE_ENCODER_RNN_NUM_LAYERS = 1;
local SENTENCE_ENCODER_RNN_DIM = 256;
local SENTENCE_ACCUMULATOR_RNN_NUM_LAYERS = 1;
local SENTENCE_ACCUMULATOR_RNN_DIM = 256;
local DROPOUT = 0.3;
local NUM_EPOCHS = 10;
local LR = 0.001;
local CUDA_DEVICE = 0;

{
  "train_data_path": TRAIN_DATA_PATH,
  "validation_data_path": VAL_DATA_PATH,
  "datasets_for_vocab_creation": ["train"],
  "dataset_reader": {
      "max_sentences_count": MAX_SENTENCES_COUNT,
      "sentence_max_tokens": SENTENCE_MAX_TOKENS,
      "type": READER,
      "lowercase": LOWERCASE,
      "tokenizer": {
        "type": "subword",
        "model_path": BPE_MODEL_PATH
      }
  },
  "vocabulary": {
    "max_vocab_size": VOCAB_SIZE
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["source_sentences", "num_fields"]],
    "batch_size": BATCH_SIZE,
    "padding_noise": 0.0,
    "cache_instances": true
  },
  "model": {
    "type": "summarunner",
    "source_embedder": {
      "type": "basic",
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": EMBEDDING_DIM
        }
      }
    },
    "sentence_encoder": {
      "type": "lstm",
      "num_layers": SENTENCE_ENCODER_RNN_NUM_LAYERS,
      "input_size": EMBEDDING_DIM,
      "hidden_size": SENTENCE_ENCODER_RNN_DIM,
      "bidirectional": true
    },
    "sentence_accumulator": {
      "type": "lstm",
      "num_layers": SENTENCE_ACCUMULATOR_RNN_NUM_LAYERS,
      "input_size": SENTENCE_ENCODER_RNN_DIM * 2,
      "hidden_size": SENTENCE_ACCUMULATOR_RNN_DIM,
      "bidirectional": true
    },
    "dropout": DROPOUT,
    "use_novelty": false,
    "use_output_bias": false,
    "use_salience": false,
    "use_pos_embedding": false
  },
  "trainer": {
    "num_epochs": NUM_EPOCHS,
    "grad_norm": 2.0,
    "cuda_device": CUDA_DEVICE,
    "patience": 1,
    "shuffle": true,
    "optimizer": {
      "type": "adam",
      "lr": LR
    }
  }
}
