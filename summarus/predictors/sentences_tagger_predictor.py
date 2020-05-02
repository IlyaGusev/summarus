from typing import List

import torch
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register('sentences_tagger')
class SentencesTaggerPredictor(Predictor):
    def __init__(self, model: Model,
                 dataset_reader: DatasetReader, top_n=3,
                 border=None, detokenize_subwords=True) -> None:
        super().__init__(model, dataset_reader)
        self._top_n = top_n
        self._border = border
        self._detokenize_subwords = detokenize_subwords

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentences = json_dict["source_sentences"]
        return self._dataset_reader.text_to_instance(sentences)

    def _process_output(self, instance, output) -> str:
        proba = torch.sigmoid(torch.Tensor(output["predicted_tags"])).tolist()
        sorted_proba = list(sorted(enumerate(proba), key=lambda x: -x[1]))
        if self._top_n is not None and self._border is None:
            indices = sorted_proba[:self._top_n]
        elif self._border is not None:
            indices = [(i, p) for i, p in sorted_proba if p > self._border]
            if self._top_n is not None:
                indices = indices[:self._top_n]
            if not indices:
                indices = sorted_proba[:1]
        indices.sort()

        sentences = instance["source_sentences"]
        hyp = [[t.text for t in sentences[i].tokens[1:-1]] for i, _ in indices]
        if self._detokenize_subwords:
            hyp = ["".join(s).replace("▁", " ").strip() for s in hyp]
        else:
            hyp = [" ".join(s) for s in hyp]
        hyp = " ".join(hyp).strip()
        return hyp

    @overrides
    def predict_instance(self, instance: Instance) -> str:
        output = self._model.forward_on_instance(instance)
        return self._process_output(instance, output)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[str]:
        outputs = self._model.forward_on_instances(instances)
        return [self._process_output(instance, output) for instance, output in zip(instances, outputs)]

    @overrides
    def dump_line(self, outputs: str) -> str:
        return outputs + "\n"

