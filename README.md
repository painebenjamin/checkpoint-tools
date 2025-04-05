# Checkpoint Tools

A small collection of helpful command-line tools for working with PyTorch checkpoints.

<div align="center">
    <img src="https://img.shields.io/static/v1?label=painebenjamin&message=checkpoint-tools&color=00519c&logo=github" alt="painebenjamin - checkpoint-tools">
    <img src="https://img.shields.io/github/stars/painebenjamin/checkpoint-tools?style=social" alt="stars - checkpoint-tools">
    <img src="https://img.shields.io/github/forks/painebenjamin/checkpoint-tools?style=social" alt="forks - checkpoint-tools"><br />
    <a href="https://github.com/painebenjamin/checkpoint-tools/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache-00519c" alt="License"></a>
    <a href="https://pypi.org/project/checkpoint-tools"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/checkpoint-tools?color=00519c"></a>
    <a href="https://pypistats.org/packages/checkpoint-tools"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/checkpoint-tools?logo=python&logoColor=white&color=00519c"></a>
</div>

# Installation

```sh
pip install checkpoint-tools
```

# Available Commands

See all with `checkpoint-tools --help`

## `metadata`

```
Usage: checkpoint-tools metadata [OPTIONS] INPUT_FILE

  Print metadata of a SafeTensors checkpoint.

Options:
  --help  Show this message and exit.
```

## `convert`

```
Usage: checkpoint-tools convert [OPTIONS] INPUT_FILE

  Convert a PyTorch/SafeTensors checkpoint to SafeTensors format, optionally
  changing the precision of the (floating point) tensors.

  Writes the file with a suffix appropriate for the precision of the tensors
  in the state dictionary.

Options:
  --float8-e5m2-fn-uz           Convert all floating point tensors to
                                float8-e5m2-fn-uz (5 exponent bits, 2 mantissa
                                bits, finite numbers only, no negative zero)
  --float8-e5m2                 Convert all floating point tensors to
                                float8-e5m2 (5 exponent bits, 2 mantissa bits)
  --float8-e4m3-fn-uz           Convert all floating point tensors to
                                float8-e4m3-fn-uz (4 exponent bits, 3 mantissa
                                bits, finite numbers only, no negative zero)
  --float8-e4m3-fn              Convert all floating point tensors to
                                float8-e4m3-fn (4 exponent bits, 3 mantissa
                                bits, finite numbers only)
  --bfloat16                    Convert all floating point tensors to bfloat16
  --float16                     Convert all floating point tensors to float16
  --full                        Leave all tensors as full precision
  --replace-key TEXT            Keys to replace, use `:` to separate old and
                                new key parts
  --ignore-key TEXT             Keys to ignore
  --overwrite / --no-overwrite  Overwrite output file if it exists
  --name TEXT                   Output file name
  --help                        Show this message and exit.
```

## `convert-to-diffusers`

```
Usage: checkpoint-tools convert-to-diffusers [OPTIONS] INPUT_FILE

  Convert a non-diffusers PyTorch/SafeTensors checkpoint to Diffusers format
  in SafeTensors.

  Writes the file with a suffix appropriate for the precision of the tensors
  in the state dictionary.

  Supported model types:

      Stable Diffusion 1.5

      Stable Diffusion XL

      Stable Diffusion 3.5

      FLUX.Dev

      FLUX.Schnell

Options:
  --name TEXT                   Output file name
  --model-type TEXT             Model type, default inferred from state
                                dictionary
  --int8                        Quantize all floating point tensors to 8-bit
                                integer using bitsandbytes
  --nf4                         Quantize all floating point tensors to
                                normalized float4 using bitsandbytes
  --float8-e5m2-fn-uz           Convert all floating point tensors to
                                float8-e5m2-fn-uz (5 exponent bits, 2 mantissa
                                bits, finite numbers only, no negative zero)
  --float8-e5m2                 Convert all floating point tensors to
                                float8-e5m2 (5 exponent bits, 2 mantissa bits)
  --float8-e4m3-fn-uz           Convert all floating point tensors to
                                float8-e4m3-fn-uz (4 exponent bits, 3 mantissa
                                bits, finite numbers only, no negative zero)
  --float8-e4m3-fn              Convert all floating point tensors to
                                float8-e4m3-fn (4 exponent bits, 3 mantissa
                                bits, finite numbers only)
  --bfloat16                    Convert all floating point tensors to bfloat16
  --float16                     Convert all floating point tensors to float16
  --full                        Leave all tensors as full precision
  --replace-key TEXT            Keys to replace, use `:` to separate old and
                                new key parts
  --ignore-key TEXT             Keys to ignore
  --overwrite / --no-overwrite  Overwrite output file if it exists
  --name TEXT                   Output file name
  --help                        Show this message and exit.
```

## `combine`

```
Usage: checkpoint-tools combine [OPTIONS] [INPUT_FILES]...

  Combine multiple checkpoints into a single checkpoint.

Options:
  --float8-e5m2-fn-uz           Convert all floating point tensors to
                                float8-e5m2-fn-uz (5 exponent bits, 2 mantissa
                                bits, finite numbers only, no negative zero)
  --float8-e5m2                 Convert all floating point tensors to
                                float8-e5m2 (5 exponent bits, 2 mantissa bits)
  --float8-e4m3-fn-uz           Convert all floating point tensors to
                                float8-e4m3-fn-uz (4 exponent bits, 3 mantissa
                                bits, finite numbers only, no negative zero)
  --float8-e4m3-fn              Convert all floating point tensors to
                                float8-e4m3-fn (4 exponent bits, 3 mantissa
                                bits, finite numbers only)
  --bfloat16                    Convert all floating point tensors to bfloat16
  --float16                     Convert all floating point tensors to float16
  --full                        Leave all tensors as full precision
  --replace-key TEXT            Keys to replace, use `:` to separate old and
                                new key parts
  --ignore-key TEXT             Keys to ignore
  --overwrite / --no-overwrite  Overwrite output file if it exists
  --name TEXT                   Output file name
  --help                        Show this message and exit.
```
