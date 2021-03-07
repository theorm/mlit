import copy

import torch
from transformers import T5ForConditionalGeneration, T5PreTrainedModel

from .decoder import (
    T5DecoderInferenceSessionWrapper,
    T5DecoderNoHistoryInferenceSessionWrapper,
)
from .encoder import T5EncoderInferenceSessionWrapper


class OnnxT5LMHeadModelNoHistory(T5ForConditionalGeneration):
    def __init__(self, config, encoder_session, decoder_session):
        T5PreTrainedModel.__init__(self, config)  # type: ignore

        self.model_dim = config.d_model

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5EncoderInferenceSessionWrapper(
            encoder_session, encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5DecoderNoHistoryInferenceSessionWrapper(
            decoder_session, decoder_config)

        # LM head is already integrated into decoder.
        # set it to pass through and disable all logit math in the original model
        self.lm_head = lambda x: x
        self.config.tie_word_embeddings = False  # type: ignore

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @property
    def device(self):
        return torch.device('cpu')


class OnnxT5LMHeadModel(T5ForConditionalGeneration):
    def __init__(self, config, encoder_session, decoder_session_first_step, decoder_session):
        T5PreTrainedModel.__init__(self, config)  # type: ignore

        self.model_dim = config.d_model

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5EncoderInferenceSessionWrapper(
            encoder_session, encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5DecoderInferenceSessionWrapper(
            decoder_session_first_step, decoder_session, decoder_config)

        # LM head is already integrated into decoder.
        # set it to pass through and disable all logit math in the original model
        self.lm_head = lambda x: x
        self.config.tie_word_embeddings = False  # type: ignore

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @property
    def device(self):
        return torch.device('cpu')
