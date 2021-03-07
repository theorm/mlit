# ML inference tools

## Requirements

For model export `onnx` package is required.

## Convert to ONNX

Below are some examples:

Convert `t5-small`:

```
PYTHONPATH=. python mlit to-onnx --model-type t5 --model-name t5-small --export-dir tmp
```

Check that it is working:

```
PYTHONPATH=. python mlit inference --model-name t5-small --base-dir tmp --model-input "translate English to French: How does this model work?" --model-type t5
```

Convert custom checkpoint:

```
PYTHONPATH=. python mlit to-onnx --model-type t5 --model-name "../my_custom_model" --export-dir tmp
```

Check that it is working:

```
PYTHONPATH=. python mlit inference --model-name my_custom_model --base-dir tmp --model-input "translate English to French: How does this model work?" --model-type t5 --tokenizer-name "t5-small"
```
