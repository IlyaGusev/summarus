from typing import List

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides


class SummaryPredictor(Predictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 fix_subwords=True) -> None:
        super().__init__(model, dataset_reader)
        self._fix_subwords = fix_subwords

    def _process_output(self, output) -> str:
        tokens = output["predicted_tokens"][0]
        text = "".join(tokens).replace("▁", " ").strip() if self._fix_subwords else " ".join(tokens).strip()
        return text

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)

    @overrides
    def predict_instance(self, instance: Instance) -> str:
        output = self._model.forward_on_instance(instance)
        return self._process_output(output)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[str]:
        outputs = self._model.forward_on_instances(instances)
        return [self._process_output(output) for output in outputs]

    @overrides
    def dump_line(self, output: str) -> str:
        return output + "\n"


@Predictor.register('words_summary')
class WordsSummaryPredictor(SummaryPredictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader, fix_subwords=False)


@Predictor.register('subwords_summary')
class SubwordsSummaryPredictor(SummaryPredictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader, fix_subwords=True)
