from argparse import ArgumentParser, Namespace
from enum import Enum
import logging

from transformers import PreTrainedTokenizerFast, pipeline


from mlit.models.t5.helpers.loader import load_inference_model, load_model as load_model_t5
from mlit.onnx_support.export import export_config, export_model

logger = logging.getLogger(__name__)


class Task(Enum):
    TO_ONNX = 'to-onnx'
    INFERENCE = 'inference'

    def __str__(self):
        return self.value


class ModelType(Enum):
    T5 = 't5'

    def __str__(self):
        return self.value


def to_onnx(args: Namespace):
    model_type = ModelType(args.model_type)
    helpers = []
    if model_type == ModelType.T5:
        helpers, config = load_model_t5(
            args.model_name_or_path,
            args.config_name,
            args.cache_dir
        )
    for helper in helpers:
        file_paths = export_model(
            helper,
            args.export_dir,
            args.do_quantize
        )
        logger.info(f'Created files: {file_paths}')
    export_config(
        config, args.export_dir, helpers[0].name
    )


def inference(args: Namespace):
    model_type = ModelType(args.model_type)
    if model_type == ModelType.T5:
        model = load_inference_model(
            args.model_name,
            args.base_dir,
            args.model_subtype,
            args.quantized
        )
        # TODO: temporary
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            args.tokenizer_name or args.model_name)
        model.__class__.__name__ = 'T5ForConditionalGeneration'
        pl = pipeline(
            'text2text-generation',
            model=model,
            tokenizer=tokenizer
        )
        result = pl(args.model_input, use_cache=True)
        print(result)


def get_subparser_to_onnx(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--model-type', dest='model_type', required=True, type=str,
                        choices=list(map(lambda x: x.value, ModelType)), help='Model type')
    parser.add_argument('--model-name', dest='model_name_or_path', required=True,
                        type=str, help='Model name or path')
    parser.add_argument('--export-dir', dest='export_dir', type=str, required=True,
                        help='Base export directory')
    parser.add_argument('--cache-dir', dest='cache_dir', type=str, required=False,
                        default=None, help='Transformers cache directory')
    parser.add_argument('--config-name', dest='config_name', type=str, required=False,
                        default=None, help='Optional config name (if different from model)')
    parser.add_argument('--quantize', dest='do_quantize', type=bool, required=False,
                        default=True, help='Quantize or not')

    parser.set_defaults(func=to_onnx)
    return parser


def get_subparser_inference(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--model-name', dest='model_name', required=True,
                        type=str, help='Model name')
    parser.add_argument('--base-dir', dest='base_dir', type=str, required=True,
                        help='Base export directory')
    parser.add_argument('--model-type', dest='model_type', required=True, type=str,
                        choices=list(map(lambda x: x.value, ModelType)), help='Model type')
    parser.add_argument('--model-subtype', dest='model_subtype', type=str, default='history',
                        choices=['history', 'no_history'], help='Model type')
    parser.add_argument('--quantized', dest='quantized', type=bool, required=False,
                        default=True, help='Quantized or not')
    parser.add_argument('--tokenizer-name', dest='tokenizer_name', type=str, default=None,
                        required=False, help='Optional tokenizer name (model name by default)')
    parser.add_argument('--model-input', dest='model_input', type=str, required=True,
                        help='Model input')

    parser.set_defaults(func=inference)

    return parser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='ML Inference tools CLI.',
    )

    subparsers = parser.add_subparsers(help='sub-command help')

    subparser = subparsers.add_parser(str(Task.TO_ONNX), help='onnx help')
    get_subparser_to_onnx(subparser)

    subparser = subparsers.add_parser(
        str(Task.INFERENCE), help='inference help')
    get_subparser_inference(subparser)

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    args.func(args)
