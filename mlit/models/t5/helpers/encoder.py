import random
from typing import TYPE_CHECKING, Tuple

import torch
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Stack

from mlit.models.common import InputDescription, OnnxModelConverterHelper

if TYPE_CHECKING:
    from onnxruntime import InferenceSession
    import torch.nn


class T5EncoderDescription(OnnxModelConverterHelper[T5Stack]):
    '''
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L836-L850
    '''
    subname = 'encoder'
    inputs = {
        'input_ids': InputDescription(
            dynamic_axis={0: 'batch_size', 1: 'encoder_seq_len'}
        ),
        'attention_mask': InputDescription(
            dynamic_axis={0: 'batch_size', 1: 'encoder_seq_len'}
        )
    }
    outputs = {
        'last_hidden_state': InputDescription(
            dynamic_axis={0: 'batch_size', 1: 'input_seq_len'}
        )
    }
    default_forward_args = {
        'encoder_hidden_states': None,
        'encoder_attention_mask': None,
        'inputs_embeds': None,
        'head_mask': None,
        'encoder_head_mask': None,
        'past_key_values': None,
        'use_cache': False,
        'output_attentions': False,
        'output_hidden_states': False,
        'return_dict': True
    }

    def __init__(self, encoder: T5Stack):
        super().__init__(encoder)

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        return [result.last_hidden_state]

    def sample_inputs(
        self,
        batch_size: int = 2,
        sequence_length: int = 50
    ) -> Tuple:

        vocab_size: int = self.model.config.vocab_size  # type: ignore

        input_ids = torch.randint(low=0,
                                  high=vocab_size - 1,
                                  size=(batch_size, sequence_length),
                                  dtype=torch.int64)

        attention_mask = torch.ones(
            [batch_size, sequence_length],
            dtype=torch.int64)

        if sequence_length >= 2:
            # mask one word in a random position
            padding_position = random.randint(1, sequence_length - 1)
            attention_mask[:, padding_position:] = 0

        return (
            input_ids,
            attention_mask
        )


class T5EncoderInferenceSessionWrapper(T5Stack):
    _onnx_model: 'InferenceSession'

    def __init__(self, onnx_model: 'InferenceSession', config):
        super(T5Stack, self).__init__(config)
        self._onnx_model = onnx_model

    def forward(self,
                input_ids=None,
                attention_mask=None,
                return_dict=None,
                **kwargs):
        [hidden_states] = self._onnx_model.run(
            None,
            {
                'input_ids': input_ids.cpu().numpy(),
                'attention_mask': attention_mask.cpu().numpy()
            }
        )
        hidden_states = torch.FloatTensor(hidden_states)

        if not return_dict:
            return tuple(hidden_states)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
        )
