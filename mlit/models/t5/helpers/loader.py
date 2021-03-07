
from transformers.configuration_utils import PretrainedConfig
from mlit.models.t5.helpers.model import OnnxT5LMHeadModel, OnnxT5LMHeadModelNoHistory
from torch.nn.modules.linear import Linear
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from mlit.models.t5.helpers.decoder import T5DecoderFirstStepHistoryDescription, T5DecoderHistoryDescription, T5DecoderNoHistoryDescription
from mlit.models.t5.helpers.encoder import T5EncoderDescription
from transformers import T5ForConditionalGeneration
from mlit.models.common import OnnxModelConverterHelper
from typing import List, Literal, Optional, Tuple, cast
import os
from onnxruntime import InferenceSession


def load_model(
    name_or_path: str,
    config_name: str = None,
    cache_dir: str = None
) -> Tuple[List[OnnxModelConverterHelper], T5Config]:

    model_config: Optional[T5Config] = None
    if config_name is not None:
        model_config = cast(T5Config, T5Config.from_pretrained(config_name))

    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        name_or_path, config=model_config, cache_dir=cache_dir).eval()  # type: ignore

    encoder: T5Stack = model.encoder  # type: ignore
    decoder: T5Stack = model.decoder  # type: ignore
    lm_head: Linear = model.lm_head  # type: ignore
    config: T5Config = model.config  # type: ignore

    return [
        T5EncoderDescription(encoder),
        T5DecoderNoHistoryDescription(decoder, lm_head),
        T5DecoderFirstStepHistoryDescription(decoder, lm_head),
        T5DecoderHistoryDescription(decoder, lm_head)
    ], config


def load_inference_model(
    name: str,
    base_dir: str,
    model_type: Literal['no_history', 'history'],
    quantized: bool
) -> T5ForConditionalGeneration:
    config_location = os.path.join(
        base_dir, name, 'transformers_config')
    config = T5Config.from_pretrained(config_location)

    if model_type == 'no_history':
        if quantized:
            encoder_path = os.path.join(
                base_dir, name, f'{T5EncoderDescription.subname}.onnx')
            decoder_path = os.path.join(
                base_dir, name, f'{T5DecoderNoHistoryDescription.subname}.onnx')
        else:
            encoder_path = os.path.join(
                base_dir, name, 'quantized', f'{T5EncoderDescription.subname}.onnx')
            decoder_path = os.path.join(
                base_dir, name, 'quantized', f'{T5DecoderNoHistoryDescription.subname}.onnx')

        return OnnxT5LMHeadModelNoHistory(
            config,
            InferenceSession(encoder_path),
            InferenceSession(decoder_path)
        )
    elif model_type == 'history':
        if quantized:
            encoder_path = os.path.join(
                base_dir, name, f'{T5EncoderDescription.subname}.onnx')
            decoder_first_step_path = os.path.join(
                base_dir, name, f'{T5DecoderFirstStepHistoryDescription.subname}.onnx')
            decoder_path = os.path.join(
                base_dir, name, f'{T5DecoderHistoryDescription.subname}.onnx')
        else:
            encoder_path = os.path.join(
                base_dir, name, 'quantized', f'{T5EncoderDescription.subname}.onnx')
            decoder_first_step_path = os.path.join(
                base_dir, name, 'quantized', f'{T5DecoderFirstStepHistoryDescription.subname}.onnx')
            decoder_path = os.path.join(
                base_dir, name, 'quantized', f'{T5DecoderHistoryDescription.subname}.onnx')

        return OnnxT5LMHeadModel(
            config,
            InferenceSession(encoder_path),
            InferenceSession(decoder_first_step_path),
            InferenceSession(decoder_path)
        )
