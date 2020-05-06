from typing import List

import torch
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides


class SummarySentencesPredictor(Predictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 top_n=3,
                 border=None,
                 fix_subwords=True) -> None:
        super().__init__(model, dataset_reader)
        self._top_n = top_n
        self._border = border
        self._fix_subwords = fix_subwords

    def _process_output(self, instance, output) -> str:
        proba = torch.sigmoid(torch.Tensor(output["predicted_tags"])).tolist()
        sorted_proba = list(sorted(enumerate(proba), key=lambda x: -x[1]))
        has_top_n = self._top_n is not None
        has_border = self._border is not None
        assert has_top_n or has_border
        if has_top_n and not has_border:
            indices = sorted_proba[:self._top_n]
        else:
            indices = [(i, p) for i, p in sorted_proba if p > self._border]
            if self._top_n is not None:
                indices = indices[:self._top_n]
            if not indices:
                indices = sorted_proba[:1]
        indices.sort()

        sentences = instance["source_sentences"]
        hyp = [[t.text for t in sentences[i].tokens[1:-1]] for i, _ in indices]
        hyp = ["".join(s).replace("▁", " ").strip() if self._fix_subwords else " ".join(s) for s in hyp]
        hyp = " ".join(hyp).strip()
        return hyp

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)

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


@Predictor.register('words_summary_sentences')
class WordsSummarySentencesPredictor(SummarySentencesPredictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 top_n=3,
                 border=None) -> None:
        super().__init__(model, dataset_reader, top_n, border, fix_subwords=False)


@Predictor.register('subwords_summary_sentences')
class SubwordsSummarySentencesPredictor(SummarySentencesPredictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 top_n=3,
                 border=None) -> None:
        super().__init__(model, dataset_reader, top_n, border, fix_subwords=True)
