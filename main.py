import os
import logging

# from rouge import Rouge
from pyrouge import Rouge155
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.training.trainer import Trainer
from allennlp.models.model import Model
from allennlp.predictors.simple_seq2seq import SimpleSeq2SeqPredictor

from summarus.seq2seq import Seq2Seq
from summarus.datasets.cnn_dailymail_reader import CNNDailyMailReader

train_urls = "/data/cnn_dailymail/all_train.txt"
val_urls = "/data/cnn_dailymail/all_val.txt"
test_urls = "/data/cnn_dailymail/all_test.txt"
cnn_dir = "/data/cnn_dailymail/cnn_stories_tokenized"
dm_dir = "/data/cnn_dailymail/dm_stories_tokenized"
train_cache = "/data/cnn_dailymail/train.pickle"
val_cache = "/data/cnn_dailymail/val.pickle"


def make_html_safe(s):
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def make_vocab(vocabulary_path, separate_namespaces=False):
    reader = CNNDailyMailReader(cnn_tokenized_dir=cnn_dir, dm_tokenized_dir=dm_dir,
                                separate_namespaces=separate_namespaces)
    train_dataset = reader.read(train_urls)
    val_dataset = reader.read(val_urls)
    test_dataset = reader.read(test_urls)
    vocabulary = Vocabulary.from_instances(test_dataset)
    vocabulary.extend_from_instances(Params({}), val_dataset)
    vocabulary.extend_from_instances(Params({}), train_dataset)
    vocabulary.save_to_files(vocabulary_path)
    return vocabulary


def train(model_name):
    models_path = "models"
    model_path = os.path.join(models_path, model_name)
    vocabulary_path = os.path.join(model_path, "vocabulary")
    params_path = os.path.join(model_path, "config.json")
    params = Params.from_file(params_path)

    if os.path.exists(vocabulary_path):
        vocabulary = Vocabulary.from_files(vocabulary_path, )
    else:
        vocabulary = make_vocab(vocabulary_path)

    reader = CNNDailyMailReader.from_params(params.pop("reader"), cnn_tokenized_dir=cnn_dir, dm_tokenized_dir=dm_dir)
    train_dataset = reader.read(train_urls)
    val_dataset = reader.read(val_urls)

    model_params = params.pop("model")
    model_params.pop("type")
    model = Seq2Seq.from_params(model_params, vocab=vocabulary)
    print(model)
    print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    iterator = DataIterator.from_params(params.pop('iterator'))
    iterator.index_with(vocabulary)
    trainer = Trainer.from_params(model, model_path, iterator,
                                  train_dataset, val_dataset, params.pop('trainer'))
    trainer.train()


def rouge_log(results_dict):
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x,y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str)


def evaluate(model_name):
    models_path = "models"
    model_path = os.path.join(models_path, model_name)
    params_path = os.path.join(model_path, "config.json")

    params = Params.from_file(params_path)
    reader = CNNDailyMailReader.from_params(params.pop("reader"), cnn_tokenized_dir=cnn_dir, dm_tokenized_dir=dm_dir)

    model = Model.load(params, model_path, cuda_device=0)
    print(model)
    print("Trainable params count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    eval_dir = os.path.join(model_path, "eval")
    ref_dir = os.path.join(eval_dir, "ref")
    hyp_dir = os.path.join(eval_dir, "hyp")
    src_dir = os.path.join(eval_dir, "src")

    count = 0
    predictor = SimpleSeq2SeqPredictor(model, reader)
    for article, reference_sents in reader.parse_files(val_urls):
        decoded_words = predictor.predict(article)["predicted_tokens"]
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")
            except ValueError:
                fst_period_idx = len(decoded_words)
            sent = decoded_words[:fst_period_idx + 1]
            decoded_words = decoded_words[fst_period_idx + 1:]
            decoded_sents.append(' '.join(sent))
        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        reference_sents = [make_html_safe(w) for w in reference_sents]

        print("Article: ", article)
        print("Abstract: ", reference_sents)
        print("Pred abstract: ", decoded_sents)

        ref_path = os.path.join(ref_dir, str(count) + "_reference.txt")
        hyp_path = os.path.join(hyp_dir, str(count) + "_decoded.txt")
        src_path = os.path.join(src_dir, str(count) + ".txt")
        with open(ref_path, "w", encoding="utf-8") as w:
            for idx, sent in enumerate(reference_sents):
                w.write(sent) if idx == len(reference_sents) - 1 else w.write(sent + "\n")
        with open(hyp_path, "w", encoding="utf-8") as w:
            for idx, sent in enumerate(decoded_sents):
                w.write(sent) if idx == len(decoded_sents) - 1 else w.write(sent + "\n")
        with open(src_path, "w", encoding="utf-8") as w:
            w.write(article)
        count += 1

    r = Rouge155(rouge_dir="/home/yallen/ROUGE-1.5.5")
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = hyp_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    scores = r.output_to_dict(rouge_results)
    rouge_log(scores)


evaluate("single_layer_lstm_external_vocab")
