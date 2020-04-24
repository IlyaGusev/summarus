from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register('sentences-tagger')
class SentencesTaggerPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, source_sentences: str) -> JsonDict:
        return self.predict_json({"source_sentences": source_sentences})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentences = json_dict["source_sentences"]
        return self._dataset_reader.text_to_instance(sentences)

