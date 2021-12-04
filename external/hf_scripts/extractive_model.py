import copy

import torch
from torch.nn.functional import pad
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification


class ModelForSentencesClassificationConfig(PretrainedConfig):
    model_type = "model-for-sentences-classification"
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

AutoConfig.register("model-for-sentences-classification", ModelForSentencesClassificationConfig)

class ModelForSentencesClassification(PreTrainedModel):
    config_class = ModelForSentencesClassificationConfig
    base_model_prefix = "model_for_sentences_classification"

    def __init__(
        self,
        config=None,
        tokens_model=None,
        sentences_model=None
    ):
        assert config is not None or (
            tokens_model is not None and sentences_model is not None
        ), "Either a configuration or both models has to be provided"
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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        assert self.config.sep_token_id >= 0
        assert self.config.max_sentences_count > 0

        batch_size = input_ids.size(0)
        sep_token_id = self.config.sep_token_id
        max_sentences_count = self.config.max_sentences_count

        sep_indices = input_ids.new_zeros((batch_size, max_sentences_count))
        for i, sample_ids in enumerate(input_ids):
            ids = (sample_ids == sep_token_id).nonzero().squeeze(1)
            ids = ids[:max_sentences_count]
            sep_indices[i, :ids.size(0)] = ids
        mask_sep = (sep_indices != 0).long()

        outputs = self.tokens_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state
        sentences_states = last_hidden_state[torch.arange(batch_size).unsqueeze(1), sep_indices]
        sentences_states = sentences_states * mask_sep[:, :, None].float()

        outputs = self.sentences_model(
            inputs_embeds=sentences_states,
            attention_mask=mask_sep,
            labels=labels
        )
        return outputs

    @classmethod
    def from_parts_pretrained(cls, tokens_model_name, sentences_model_config):
        tokens_model = AutoModel.from_pretrained(tokens_model_name)
        config = ModelForSentencesClassificationConfig.from_configs(tokens_model.config, sentences_model_config)
        return cls(tokens_model=tokens_model, config=config)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)
