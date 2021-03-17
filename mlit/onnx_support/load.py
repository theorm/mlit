
import logging
import io
from typing import List, Optional, Union
import onnxruntime
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = logging.getLogger(__name__)


def load_inference_session(
    file_paths: Union[str, List[str]],
    options: Optional[onnxruntime.SessionOptions] = None,
    use_threading=True
):

    def _create_session(file_path: str) -> onnxruntime.InferenceSession:
        if file_path.endswith('.zst'):
            from pyzstd import decompress_stream

            logger.debug(f'Reading compressed model file: {file_path}')
            with io.open(file_path, 'rb') as ifh:
                with io.BytesIO() as bo:
                    decompress_stream(ifh, bo)
                    model_bytes = bo.getvalue()
            logger.debug(
                f'Creating session from compressed model file: {file_path}')
            sess = onnxruntime.InferenceSession(model_bytes, options)
            # Free memory
            # https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/onnxruntime_inference_collection.py#L270
            sess._model_bytes = None
            logger.debug(
                f'Session created from compressed model file: {file_path}')
            return sess
        else:
            logger.debug(f'Creating session from model file: {file_path}')
            sess = onnxruntime.InferenceSession(file_path, options)
            logger.debug(f'Session created from model file: {file_path}')
            return sess

    session_file_paths = [file_paths] if isinstance(
        file_paths, str) else file_paths

    if use_threading and len(session_file_paths) > 1:
        with ThreadPoolExecutor() as executor:
            futures = []
            for idx, path in enumerate(session_file_paths):
                futures.append(executor.submit(
                    lambda p: (idx, _create_session(p)), p=path))
            sessions = [future.result() for future in as_completed(futures)]
            sessions = sorted(sessions, key=lambda s: s[0])
            sessions = [s[1] for s in sessions]
    else:
        sessions = [_create_session(p) for p in session_file_paths]

    if isinstance(file_paths, str):
        return sessions[0]
    else:
        return sessions
