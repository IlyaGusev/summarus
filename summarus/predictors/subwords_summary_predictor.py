from typing import List

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides

@Predictor.register('subwords_summary')
class SubwordsSummaryPredictor(Predictor):
    def _process_outputs(self, outputs) -> str:
        tokens = outputs["predicted_tokens"]
        text = "".join(tokens).replace("▁", " ").strip()
        return text

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
       source = json_dict["source"]
       return self._dataset_reader.text_to_instance(source)

    @overrides
    def predict_instance(self, instance: Instance) -> str:
        outputs = self._model.forward_on_instance(instance)
        return self._process_outputs(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[str]:
        outputs = self._model.forward_on_instances(instances)
        return [self._process_outputs(output) for output in outputs]

    @overrides
    def dump_line(self, outputs: str) -> str:
        return outputs + "\n"

