
import os
import io
import logging
from typing import List, Literal, Optional, TYPE_CHECKING, Tuple, Union, cast
from concurrent.futures import ThreadPoolExecutor, as_completed
from onnxruntime import InferenceSession, SessionOptions
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

logger = logging.getLogger(__name__)


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
    quantized: bool,
    use_threading: bool = True,
    use_compressed_files: bool = False,
    options: Optional[SessionOptions] = None
) -> T5ForConditionalGeneration:
    config_location = os.path.join(
        base_dir, name, 'transformers_config')
    config = T5Config.from_pretrained(config_location)

    base_path = [base_dir, name]

    file_suffix = 'onnx' if not use_compressed_files else 'onnx.zst'

    def construct_path(filename: str) -> str:
        parts = base_path + [filename]
        if quantized:
            parts = _as_quantized(parts)
        return os.path.join(*parts)

    if model_type == 'no_history':
        encoder_path = construct_path(
            f'{T5EncoderDescription.subname}.{file_suffix}')
        decoder_path = construct_path(
            f'{T5DecoderNoHistoryDescription.subname}.{file_suffix}')

        klass = OnnxT5LMHeadModelNoHistory
        klass_config = config
        klass_session_paths = [encoder_path, decoder_path]

    elif model_type == 'history':
        encoder_path = construct_path(
            f'{T5EncoderDescription.subname}.{file_suffix}')
        decoder_first_step_path = construct_path(
            f'{T5DecoderFirstStepHistoryDescription.subname}.{file_suffix}')
        decoder_path = construct_path(
            f'{T5DecoderHistoryDescription.subname}.{file_suffix}')

        klass = OnnxT5LMHeadModel
        klass_config = config
        klass_session_paths = [encoder_path,
                               decoder_first_step_path, decoder_path]

    else:
        raise Exception(f'Unknown model type: {model_type}')

    def _create_session(file_path: str, options: SessionOptions) -> InferenceSession:
        if use_compressed_files:
            # import lzma
            from pyzstd import decompress_stream

            logger.debug(f'Reading compressed model file: {file_path}')
            with io.open(file_path, 'rb') as ifh:
                with io.BytesIO() as bo:
                    decompress_stream(ifh, bo)
                    model_bytes = bo.getvalue()
            # with lzma.open(file_path, 'rb') as f:
            #     model_bytes = f.read()
            logger.debug(
                f'Creating session from compressed model file: {file_path}')
            sess = InferenceSession(model_bytes, options)
            # Free memory
            # https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/onnxruntime_inference_collection.py#L270
            sess._model_bytes = None
            logger.debug(
                f'Session created from compressed model file: {file_path}')
            return sess
        else:
            logger.debug(f'Creating session from model file: {file_path}')
            sess = InferenceSession(file_path, options)
            logger.debug(f'Session created from model file: {file_path}')
            return sess

    if use_threading:
        with ThreadPoolExecutor() as executor:
            futures = []
            for idx, path in enumerate(klass_session_paths):
                futures.append(executor.submit(
                    lambda p: (idx, _create_session(p, options)), p=path))
            sessions = [future.result() for future in as_completed(futures)]
            sessions = sorted(sessions, key=lambda s: s[0])
            sessions = [s[1] for s in sessions]
    else:
        sessions = [_create_session(p, options) for p in klass_session_paths]

    return klass(klass_config, *sessions)


def load_inference_model_from_files(
    filenames: Union[Tuple[str, str], Tuple[str, str, str]],
    transformers_config_path: str,
    options: Optional[SessionOptions] = None
) -> T5ForConditionalGeneration:
    config = T5Config.from_pretrained(transformers_config_path)

    if len(filenames) == 2:
        klass = OnnxT5LMHeadModelNoHistory
    elif len(filenames) == 3:
        klass = OnnxT5LMHeadModel
    else:
        raise Exception(f'Unexpected number of filenames: {len(filenames)}')

    def _create_session(file_path: str, options: SessionOptions) -> InferenceSession:
        if file_path.endswith('.zst'):
            from pyzstd import decompress_stream

            logger.debug(f'Reading compressed model file: {file_path}')
            with io.open(file_path, 'rb') as ifh:
                with io.BytesIO() as bo:
                    decompress_stream(ifh, bo)
                    model_bytes = bo.getvalue()
            logger.debug(
                f'Creating session from compressed model file: {file_path}')
            sess = InferenceSession(model_bytes, options)
            # Free memory
            # https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/onnxruntime_inference_collection.py#L270
            sess._model_bytes = None
            logger.debug(
                f'Session created from compressed model file: {file_path}')
            return sess
        else:
            logger.debug(f'Creating session from model file: {file_path}')
            sess = InferenceSession(file_path, options)
            logger.debug(f'Session created from model file: {file_path}')
            return sess

    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, path in enumerate(filenames):
            futures.append(executor.submit(
                lambda p: (idx, _create_session(p, options)), p=path))
        sessions = [future.result() for future in as_completed(futures)]
        sessions = sorted(sessions, key=lambda s: s[0])
        sessions = [s[1] for s in sessions]

    return klass(config, *sessions)
