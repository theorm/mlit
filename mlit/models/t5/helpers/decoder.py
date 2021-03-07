from math import floor
import random
from typing import List, TYPE_CHECKING, Tuple, cast

import torch
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Stack

from mlit.models.common import InputDescription, OnnxModelConverterHelper

if TYPE_CHECKING:
    from onnxruntime import InferenceSession
    import torch.nn


class T5DecoderNoHistoryDescription(OnnxModelConverterHelper['T5Stack']):
    '''
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L836-L850
    '''
    subname = 'decoder_no_history'

    inputs = {
        'input_ids': InputDescription(
            dynamic_axis={0: 'batch_size', 1: 'decoder_seq_len'}
        ),
        'attention_mask': InputDescription(
            dynamic_axis={0: 'batch_size', 1: 'decoder_seq_len'}
        ),
        'encoder_hidden_states': InputDescription(
            dynamic_axis={0: 'batch_size', 1: 'encoder_seq_len'}
        ),
        'encoder_attention_mask': InputDescription(
            dynamic_axis={0: 'batch_size', 1: 'encoder_seq_len'}
        )
    }
    outputs = {
        'logits': InputDescription(
            dynamic_axis={0: 'batch_size', 1: 'output_seq_len'}
        )
    }
    default_forward_args = {
        'inputs_embeds': None,
        'head_mask': None,
        'encoder_head_mask': None,
        'past_key_values': None,
        'use_cache': False,
        'output_attentions': False,
        'output_hidden_states': False,
        'return_dict': True
    }

    _lm_head: torch.nn.Linear

    def __init__(self, encoder: 'T5Stack', lm_head: torch.nn.Linear):
        super().__init__(encoder)
        self._lm_head = lm_head

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)

        if self.model.config.tie_word_embeddings:
            d_model = self.model.config.d_model  # type: ignore
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            state = result.last_hidden_state * \
                (d_model ** -0.5)
        else:
            state = result.last_hidden_state

        return [self._lm_head(state)]

    def sample_inputs(
        self,
        batch_size: int = 2,
        encoder_sequence_length: int = 50,
        decoder_sequence_length: int = 20
    ) -> Tuple:

        vocab_size: int = self.model.config.vocab_size  # type: ignore
        hidden_size: int = self.model.config.d_model  # type: ignore

        input_ids = torch.randint(low=0,
                                  high=vocab_size - 1,
                                  size=(batch_size, decoder_sequence_length),
                                  dtype=torch.int64)

        attention_mask = torch.ones(
            [batch_size, decoder_sequence_length],
            dtype=torch.int64)

        if decoder_sequence_length >= 2:
            # mask one word in a random position
            padding_position = random.randint(1, decoder_sequence_length - 1)
            attention_mask[:, padding_position:] = 0

        encoder_hidden_states = torch.rand(batch_size,
                                           encoder_sequence_length,
                                           hidden_size,
                                           dtype=torch.float32)

        encoder_attention_mask = torch.ones(
            [batch_size, encoder_sequence_length],
            dtype=torch.int64)

        return (
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask
        )


class T5DecoderNoHistoryInferenceSessionWrapper(T5Stack):
    def __init__(self, onnx_model: 'InferenceSession', config):
        super(T5Stack, self).__init__(config)
        self._onnx_model = onnx_model

    def forward(self,
                input_ids=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                return_dict=None,
                **kwargs):

        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape,
                dtype=torch.int64)
        [logits] = self._onnx_model.run(
            None,
            {
                'input_ids': input_ids.cpu().numpy(),
                'attention_mask': attention_mask.cpu().numpy(),
                'encoder_hidden_states': encoder_hidden_states.cpu().numpy(),
                'encoder_attention_mask': encoder_attention_mask.cpu().numpy()
            }
        )
        logits = torch.FloatTensor(logits)

        if not return_dict:
            return tuple(logits)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=logits,
        )


class T5DecoderFirstStepHistoryDescription(T5DecoderNoHistoryDescription):
    subname = 'decoder_history_first_step'

    def __init__(self, decoder: T5Stack, lm_head: torch.nn.Linear):
        super().__init__(decoder, lm_head)

        num_decoder_layers: int = decoder.config.num_decoder_layers  # type: ignore

        # self.inputs['input_ids'].dynamic_axis = {0: 'batch_size'}

        extra_outputs = {}
        for i in range(num_decoder_layers):
            for j in range(4):
                extra_outputs[f'past_key_value_{i}_{j}'] = InputDescription(
                    dynamic_axis={0: 'batch_size', 1: 'decoder_seq_len'}
                )

        self.outputs = {
            **self.outputs,
            **extra_outputs
        }

        self.default_forward_args['use_cache'] = True

    def forward(self, *args, **kwargs):
        result = OnnxModelConverterHelper.forward(self, *args, **kwargs)

        if self.model.config.tie_word_embeddings:
            d_model = self.model.config.d_model  # type: ignore
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            state = result.last_hidden_state * (d_model ** -0.5)
        else:
            state = result.last_hidden_state

        # unravel past key values
        past_key_values = result.past_key_values
        past_key_values_outputs = []
        for i, item in enumerate(past_key_values):
            for j, inner_item in enumerate(item):
                past_key_values_outputs += [inner_item]

        return [self._lm_head(state)] + past_key_values_outputs

    def sample_inputs(
        self,
        batch_size: int = 2,
        encoder_sequence_length: int = 50,
        decoder_sequence_length: int = 20
    ) -> Tuple:
        inputs = T5DecoderNoHistoryDescription.sample_inputs(
            self, batch_size, encoder_sequence_length, decoder_sequence_length)

        a, b, c, d = inputs
        # when past key values are present, only the last input is interesting
        inputs = (a[:, -1:], b[:, -1:], c, d)

        return inputs


def from_unravelled_past_key_values(
    unraveled_past_key_values: List[torch.FloatTensor]
) -> Tuple[Tuple[torch.FloatTensor]]:
    past_key_values = {}

    for idx, kv in enumerate(unraveled_past_key_values):
        i = floor(idx / 4)
        j = idx % 4

        if i not in past_key_values:
            past_key_values[i] = {}
        past_key_values[i][j] = kv

    return tuple([
        tuple([
            past_key_values[i][j]
            for j in sorted(past_key_values[i].keys())
        ])
        for i in sorted(past_key_values.keys())
    ])


def to_unravelled_past_key_values(past_key_values):
    # unravel past key values
    past_key_values_outputs = []
    for i, item in enumerate(past_key_values):
        for j, inner_item in enumerate(item):
            past_key_values_outputs += [inner_item]
    return past_key_values_outputs


def generate_past_key_values_fields(num_layers):
    names = []
    for i in range(int(num_layers)):
        for j in range(4):
            if j < 2:
                names += [f'input_past_key_value_{i}_{j}']
            else:
                names += [f'past_key_value_{i}_{j}']
    return names


class T5DecoderHistoryDescription(T5DecoderNoHistoryDescription):
    subname = 'decoder_history'

    def __init__(self, decoder: T5Stack, lm_head: torch.nn.Linear):
        super().__init__(decoder, lm_head)

        num_decoder_layers: int = decoder.config.num_decoder_layers  # type: ignore

        # self.inputs['input_ids'].dynamic_axis = {0: 'batch_size'}

        extra_outputs = {}
        extra_inputs = {}
        for i in range(num_decoder_layers):
            for j in range(4):
                extra_outputs[f'past_key_value_{i}_{j}'] = InputDescription(
                    dynamic_axis={0: 'batch_size',
                                  2: 'decoder_seq_len' if j < 2 else 'moo'}
                )
                input_key = f'input_past_key_value_{i}_{j}' if j < 2 else f'past_key_value_{i}_{j}'
                extra_inputs[input_key] = InputDescription(
                    dynamic_axis={0: 'batch_size',
                                  2: 'decoder_seq_len' if j < 2 else 'moo'}
                )

        self.outputs = {
            **self.outputs,
            **extra_outputs
        }

        self.inputs = {
            **self.inputs,
            **extra_inputs
        }

        self.default_forward_args['use_cache'] = True
        if 'past_key_values' in self.default_forward_args:
            del self.default_forward_args['past_key_values']

    def forward(self, *args, **kwargs):

        num_decoder_layers: int = self.model.config.num_decoder_layers  # type: ignore
        offset = len(args) - (num_decoder_layers * 4)
        unraveled_past_key_values = args[offset:]

        past_key_values = from_unravelled_past_key_values(
            cast(List[torch.FloatTensor], unraveled_past_key_values))

        args = args[:offset]
        kwargs['past_key_values'] = past_key_values

        result = OnnxModelConverterHelper.forward(self, *args, **kwargs)

        if self.model.config.tie_word_embeddings:
            d_model = self.model.config.d_model  # type: ignore
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            state = result.last_hidden_state * (d_model ** -0.5)
        else:
            state = result.last_hidden_state

        return [self._lm_head(state)] + to_unravelled_past_key_values(result.past_key_values)

    def sample_inputs(
        self,
        batch_size: int = 2,
        encoder_sequence_length: int = 50,
        decoder_sequence_length: int = 20
    ) -> Tuple:
        inputs = T5DecoderNoHistoryDescription.sample_inputs(
            self, batch_size, encoder_sequence_length, decoder_sequence_length)

        num_decoder_layers: int = self.model.config.num_decoder_layers  # type: ignore
        num_heads: int = self.model.config.num_heads  # type: ignore
        d_kv: int = self.model.config.d_kv  # type: ignore

        past_key_values = []

        for i in range(num_decoder_layers):
            for j in range(4):
                if j < 2:
                    third_dim = decoder_sequence_length
                else:
                    third_dim = encoder_sequence_length

                past_key_values += [
                    torch.rand(batch_size,
                               num_heads,
                               third_dim,
                               d_kv,
                               dtype=torch.float32)
                ]
        a, b, c, d = inputs
        # when past key values are present, only the last input is interesting
        inputs = (a[:, -1:], b[:, -1:], c, d)

        return inputs + tuple(past_key_values)


class T5DecoderInferenceSessionWrapper(T5Stack):
    def __init__(self, onnx_model_first_step: 'InferenceSession', onnx_model: 'InferenceSession', config):
        super(T5Stack, self).__init__(config)
        self._onnx_model = onnx_model
        self._onnx_model_first_step = onnx_model_first_step

    def forward(self,
                input_ids=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                return_dict=None,
                **kwargs):

        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape,
                dtype=torch.int64)

        if past_key_values is not None:
            unravelled_past_key_values = to_unravelled_past_key_values(
                past_key_values)
            unravelled_past_key_value_names = generate_past_key_values_fields(
                len(unravelled_past_key_values) / 4)
            past_key_value_inputs = {
                k: v.cpu().numpy()
                for k, v in zip(unravelled_past_key_value_names, unravelled_past_key_values)
            }

            usual_inputs = {
                'input_ids': input_ids.cpu().numpy(),
                'attention_mask': attention_mask.cpu().numpy(),
                'encoder_hidden_states': encoder_hidden_states.cpu().numpy(),
                'encoder_attention_mask': encoder_attention_mask.cpu().numpy(),
            }
            results = self._onnx_model.run(
                None,
                {
                    **usual_inputs,
                    **past_key_value_inputs
                }
            )
        else:
            results = self._onnx_model_first_step.run(
                None,
                {
                    'input_ids': input_ids.cpu().numpy(),
                    'attention_mask': attention_mask.cpu().numpy(),
                    'encoder_hidden_states': encoder_hidden_states.cpu().numpy(),
                    'encoder_attention_mask': encoder_attention_mask.cpu().numpy()
                }
            )
        results = [torch.FloatTensor(i) for i in results]
        logits = results[0]
        past_key_values = results[1:]
        past_key_values = from_unravelled_past_key_values(past_key_values)

        if not return_dict:
            return tuple(logits)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=logits,
            past_key_values=past_key_values
        )
