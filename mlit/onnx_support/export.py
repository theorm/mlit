import logging
import os
import os.path
from typing import TYPE_CHECKING, Tuple, Union

import torch.onnx

from mlit.models.common import OnnxModelConverterHelper

if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


def export_config(
    config: 'PretrainedConfig',
    output_base_dir: str,
    model_name
):
    config_dir = os.path.join(
        output_base_dir, model_name, 'transformers_config')
    if not os.path.isdir(config_dir):
        os.mkdir(config_dir)

    config.save_pretrained(config_dir)


def export_model(
    model_helper: OnnxModelConverterHelper,
    output_base_dir: str,
    do_quantize: bool = False
) -> Union[str, Tuple[str, str]]:
    output_base_dir = os.path.abspath(output_base_dir)
    assert os.path.isdir(
        output_base_dir), f'No such base directory: {output_base_dir}'

    export_dir = os.path.join(output_base_dir, model_helper.name)
    if not os.path.isdir(export_dir):
        os.mkdir(export_dir)

    export_file = os.path.join(export_dir, f'{model_helper.subname}.onnx')

    args = model_helper.get_onnx_export_args()
    args['f'] = export_file
    logger.debug(f'Exporting {export_file} with {args["input_names"]}')

    torch.onnx.export(**args)

    if do_quantize:
        from mlit.onnx_support.quantize import quantize_model

        quantized_file = quantize_model(model_helper, output_base_dir)
        return (export_file, quantized_file)
    else:
        return export_file
