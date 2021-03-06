local TRAIN_DATA_PATH = std.extVar("TRAIN_DATA_PATH");
local VAL_DATA_PATH = std.extVar("VAL_DATA_PATH");
local BPE_MODEL_PATH = "https://bpe-models.s3.eu-west-3.amazonaws.com/bpe_gazeta_5k.model";
local READER = "gazeta";
local LOWERCASE = true;
local SOURCE_MAX_TOKENS = 800;
local TARGET_MAX_TOKENS = 200;
local VOCAB_SIZE = 5000;
local BATCH_SIZE = 16;
local EMBEDDING_DIM = 128;
local RNN_DIM = 256;
local RNN_NUM_LAYERS = 2;
local MAX_DECODING_STEPS = 200;
local BEAM_SIZE = 4;
local NUM_EPOCHS = 10;
local LR = 0.001;
local CUDA_DEVICE = 0;

{
  "train_data_path": TRAIN_DATA_PATH,
  "validation_data_path": VAL_DATA_PATH,
  "datasets_for_vocab_creation": ["train"],
  "dataset_reader": {
      "source_max_tokens": SOURCE_MAX_TOKENS,
      "target_max_tokens": TARGET_MAX_TOKENS,
      "tokenizer": {
        "type": "subword",
        "model_path": BPE_MODEL_PATH
      },
      "save_copy_fields": true,
      "separate_namespaces": true,
      "target_namespace": "target_tokens",
      "type": READER,
      "lowercase": LOWERCASE
  },
  "vocabulary": {
    "max_vocab_size": VOCAB_SIZE
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["source_tokens", "num_tokens"]],
    "batch_size": BATCH_SIZE,
    "padding_noise": 0.0,
    "cache_instances": false
  },
  "model": {
    "type": "custom_copynet_seq2seq",
    "target_namespace": "target_tokens",
    "source_embedder": {
      "type": "basic",
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": EMBEDDING_DIM
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "num_layers": RNN_NUM_LAYERS,
      "input_size": EMBEDDING_DIM,
      "hidden_size": RNN_DIM,
      "bidirectional": true
    },
    "tie_embeddings": false,
    "attention": {
      "type": "dot_product"
    },
    "max_decoding_steps": MAX_DECODING_STEPS,
    "beam_size": BEAM_SIZE
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
