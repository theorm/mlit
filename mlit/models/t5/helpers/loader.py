
import os
from typing import List, Literal, Optional, TYPE_CHECKING, Tuple, cast

from onnxruntime import InferenceSession
from transformers import T5Config, T5ForConditionalGeneration

from mlit.models.t5.helpers.decoder import (
    T5DecoderFirstStepHistoryDescription,
    T5DecoderHistoryDescription,
    T5DecoderNoHistoryDescription,
)
from mlit.models.t5.helpers.encoder import T5EncoderDescription
from mlit.models.t5.helpers.model import OnnxT5LMHeadModel, OnnxT5LMHeadModelNoHistory

if TYPE_CHECKING:
    from transformers.models.t5.modeling_t5 import T5Stack
    from mlit.models.common import OnnxModelConverterHelper
    from torch.nn.modules.linear import Linear


def load_model(
    name_or_path: str,
    config_name: str = None,
    cache_dir: str = None
) -> Tuple[List['OnnxModelConverterHelper'], T5Config]:

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


def _as_quantized(path_elements: List[str]) -> List[str]:
    return path_elements[:-1] + ['quantized'] + path_elements[-1:]


def load_inference_model(
    name: str,
    base_dir: str,
    model_type: Literal['no_history', 'history'],
    quantized: bool
) -> T5ForConditionalGeneration:
    config_location = os.path.join(
        base_dir, name, 'transformers_config')
    config = T5Config.from_pretrained(config_location)

    base_path = [base_dir, name]

    def construct_path(filename: str) -> str:
        parts = base_path + [filename]
        if quantized:
            parts = _as_quantized(parts)
        return os.path.join(*parts)

    if model_type == 'no_history':
        encoder_path = construct_path(f'{T5EncoderDescription.subname}.onnx')
        decoder_path = construct_path(
            f'{T5DecoderNoHistoryDescription.subname}.onnx')

        return OnnxT5LMHeadModelNoHistory(
            config,
            InferenceSession(encoder_path),
            InferenceSession(decoder_path)
        )
    elif model_type == 'history':
        encoder_path = construct_path(f'{T5EncoderDescription.subname}.onnx')
        decoder_first_step_path = construct_path(
            f'{T5DecoderFirstStepHistoryDescription.subname}.onnx')
        decoder_path = construct_path(
            f'{T5DecoderHistoryDescription.subname}.onnx')

        return OnnxT5LMHeadModel(
            config,
            InferenceSession(encoder_path),
            InferenceSession(decoder_first_step_path),
            InferenceSession(decoder_path)
        )
