import os
import os.path
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, quantize_qat, quantize_static

from mlit.models.common import OnnxModelConverterHelper


def quantize_model(
    model_helper: OnnxModelConverterHelper,
    output_base_dir: str
) -> str:
    output_base_dir = os.path.abspath(output_base_dir)
    assert os.path.isdir(
        output_base_dir), f'No such base directory: {output_base_dir}'

    export_dir = os.path.join(output_base_dir, model_helper.name)
    assert os.path.isdir(
        export_dir), f'No such export directory: {export_dir}'

    input_file = os.path.join(export_dir, f'{model_helper.subname}.onnx')
    assert os.path.isfile(
        input_file), f'No such input file found: {input_file}'

    quantize_dir = os.path.join(export_dir, 'quantized')
    if not os.path.isdir(quantize_dir):
        os.mkdir(quantize_dir)

    output_file = os.path.join(quantize_dir, os.path.basename(input_file))

    method = model_helper.onnx_quantize_method
    args = model_helper.onnx_quantize_args

    input_file = Path(input_file)
    output_file = Path(output_file)

    if method == 'quantize_static':
        out_model = quantize_static(input_file, output_file, ** args)
    elif method == 'quantize_dynamic':
        out_model = quantize_dynamic(input_file, output_file, **args)
    elif method == 'quantize_qat':
        out_model = quantize_qat(input_file, output_file, **args)
    else:
        assert False, f'Unknown quantization method: {method}'

    return output_file.as_posix()
