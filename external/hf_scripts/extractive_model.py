import copy
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification


class ModelForSentencesClassificationConfig(PretrainedConfig):
    model_type = "bert-model-for-sentences-classification"
    is_composition = True

    def __init__(self, tokens_model_config, sentences_model_config, **kwargs):
        super().__init__(**kwargs)

        tokens_model_type = tokens_model_config.pop("model_type")
        sentences_model_type = sentences_model_config.pop("model_type")

        self.tokens_model_config = AutoConfig.for_model(tokens_model_type, **tokens_model_config)
        self.sentences_model_config = AutoConfig.for_model(sentences_model_type, **sentences_model_config)

    @classmethod
    def from_configs(cls, tokens_model_config, sentences_model_config, **kwargs):
        return cls(
            tokens_model_config=tokens_model_config.to_dict(),
            sentences_model_config=sentences_model_config.to_dict(),
            **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["tokens_model_config"] = self.tokens_model_config.to_dict()
        output["sentences_model_config"] = self.sentences_model_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class ModelForSentencesClassification(PreTrainedModel):
    config_class = ModelForSentencesClassificationConfig

    def __init__(
        self,
        config=None,
        tokens_model=None,
        sentences_model=None
    ):
        assert config is not None or (
            encoder is not None and decoder is not None
        ), "Either a configuration or an Encoder and a decoder has to be provided"
        if config is None:
            config = ModelForSentencesClassificationConfig.from_configs(
                tokens_model.config, sentences_model.config
            )
        super().__init__(config)

        if tokens_model is None:
            tokens_model = AutoModel.from_config(config.tokens_model_config)

        if sentences_model is None:
            sentences_model = AutoModelForTokenClassification.from_config(config.sentences_model_config)

        self.tokens_model = tokens_model
        self.sentences_model = sentences_model

        self.tokens_model.config = self.config.tokens_model_config
        self.sentences_model.config = self.config.sentences_model_config

        self.num_labels = config.num_labels

    def forward(
        self,
        input_ids=None,
        labels=None,
        return_dict=None,
        **kwargs
    ):
        assert self.config.sep_token_id >= 0
        assert self.config.max_sentences_count > 0

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.tokens_model(
            input_ids,
            return_dict=return_dict,
            **kwargs
        )
        sequence_output = outputs[0]
        sep_indices = input_ids == self.config.sep_token_id

        batch_size = sequence_output.size(0)
        sentences_shape = (batch_size, self.config.max_sentences_count, self.config.tokens_model_config.hidden_size)
        sentences_states = sequence_output.new_zeros(sentences_shape)
        for i in range(batch_size):
            sentences_count = min(torch.sum(sep_indices[i]), self.config.max_sentences_count)
            sentences_states[i][:sentences_count] = sequence_output[i][sep_indices[i]][:sentences_count]

        outputs = self.sentences_model(inputs_embeds=sentences_states, labels=labels, return_dict=return_dict)
        return outputs

    @classmethod
    def from_parts_pretrained(cls, tokens_model_name, sentences_model_config):
        tokens_model_config = AutoConfig.from_pretrained(tokens_model_name)
        tokens_model = AutoModel.from_pretrained(tokens_model_name)
        config = ModelForSentencesClassificationConfig.from_configs(tokens_model.config, sentences_model_config)
        return cls(tokens_model=tokens_model, config=config)

