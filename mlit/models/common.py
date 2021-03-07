from dataclasses import dataclass
from typing import Any, Dict, Generic, Literal, Optional, TYPE_CHECKING, Tuple, TypeVar

import torch.nn

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

T = TypeVar('T', bound=torch.nn.Module)


@dataclass
class InputDescription:
    dynamic_axis: Optional[Dict[int, str]]


class OnnxModelConverterHelper(torch.nn.Module, Generic[T]):
    _model: T
    inputs: Dict[str, InputDescription]
    outputs: Dict[str, InputDescription]
    default_forward_args: Dict[str, Any] = {}
    subname: str = 'model'

    def __init__(self, model: T):
        super().__init__()
        self._model = model

    @property
    def model(self):
        return self._model

    @property
    def config(self) -> 'PretrainedConfig':
        return self.model.config  # type: ignore

    @property
    def name(self) -> str:
        name: str = self.model.config.name_or_path  # type: ignore
        name = name.split('/')[-1]
        return name

    def forward(self, *args, **kwargs):
        return self.model.forward(
            *args,
            **{**self.default_forward_args, **kwargs}
        )

    def sample_inputs(self) -> Tuple:
        return ()

    def get_onnx_export_args(self) -> Dict[str, Any]:
        dynamic_axes = {}
        for name, desc in self.inputs.items():
            if desc.dynamic_axis is not None:
                dynamic_axes[name] = desc.dynamic_axis
        for name, desc in self.outputs.items():
            if desc.dynamic_axis is not None:
                dynamic_axes[name] = desc.dynamic_axis

        return {
            'model': self,
            'args': self.sample_inputs(),
            'input_names': list(self.inputs.keys()),
            'output_names': list(self.outputs.keys()),
            'dynamic_axes': dynamic_axes,
            'opset_version': 12,
            'export_params': True,
            'do_constant_folding': True
        }

    @property
    def onnx_quantize_method(self) -> Literal['quantize_static', 'quantize_dynamic', 'quantize_qat']:
        return 'quantize_qat'

    @property
    def onnx_quantize_args(self) -> Dict[str, Any]:
        from onnxruntime.quantization import QuantType
        return {
            'weight_type': QuantType.QInt8,
            'op_types_to_quantize': ['MatMul']
        }
