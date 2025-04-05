#!/usr/bin/env python
import os
import click
import safetensors
import safetensors.torch

from typing import Optional, List, Any, Callable

from .util import (
    cyan,
    green,
    load_metadata,
    load_state_dict,
    get_filtered_renamed_state_dict,
    convert_state_dict_dtype,
    get_extension_for_state_dict,
    get_diffusers_state_dicts_from_checkpoint,
    quantize_state_dict_for_model
)

def precision_options(include_quantization: bool=False) -> Callable[..., Any]:
    """
    Add precision options to a command.
    """
    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn = click.option(
            "--full",
            "precision",
            flag_value="full",
            default=True,
            is_flag=True,
            help="Leave all tensors as full precision"
        )(fn)
        fn = click.option(
            "--float32",
            "precision",
            flag_value="float32",
            default=False,
            is_flag=True,
            help="Convert all floating point tensors to float32"
        )(fn)
        fn = click.option(
            "--float16",
            "precision",
            flag_value="float16",
            default=False,
            is_flag=True,
            help="Convert all floating point tensors to float16"
        )(fn)
        fn = click.option(
            "--bfloat16",
            "precision",
            flag_value="bfloat16",
            default=False,
            is_flag=True,
            help="Convert all floating point tensors to bfloat16"
        )(fn)
        fn = click.option(
            "--float8-e4m3-fn",
            "precision",
            flag_value="float8-e4m3-fn",
            default=False,
            is_flag=True,
            help="Convert all floating point tensors to float8-e4m3-fn (4 exponent bits, 3 mantissa bits, finite numbers only)"
        )(fn)
        fn = click.option(
            "--float8-e4m3-fn-uz",
            "precision",
            flag_value="float8-e4m3-fn-uz",
            default=False,
            is_flag=True,
            help="Convert all floating point tensors to float8-e4m3-fn-uz (4 exponent bits, 3 mantissa bits, finite numbers only, no negative zero)"
        )(fn)
        fn = click.option(
            "--float8-e5m2",
            "precision",
            flag_value="float8-e5m2",
            default=False,
            is_flag=True,
            help="Convert all floating point tensors to float8-e5m2 (5 exponent bits, 2 mantissa bits)"
        )(fn)
        fn = click.option(
            "--float8-e5m2-fn-uz",
            "precision",
            flag_value="float8-e5m2-fn-uz",
            default=False,
            is_flag=True,
            help="Convert all floating point tensors to float8-e5m2-fn-uz (5 exponent bits, 2 mantissa bits, finite numbers only, no negative zero)"
        )(fn)
        if include_quantization:
            fn = click.option(
                "--nf4",
                "precision",
                flag_value="nf4",
                default=False,
                is_flag=True,
                help="Quantize all floating point tensors to normalized float4 using bitsandbytes"
            )(fn)
            fn = click.option(
                "--int8",
                "precision",
                flag_value="int8",
                default=False,
                is_flag=True,
                help="Quantize all floating point tensors to 8-bit integer using bitsandbytes"
            )(fn)
        return fn
    return wrap

def checkpoint_options() -> Callable[..., Any]:
    """
    Add checkpoint options to a command.
    """
    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn = click.option("--name", type=str, default=None, help="Output file name")(fn)
        fn = click.option("--overwrite/--no-overwrite", default=False, is_flag=True, help="Overwrite output file if it exists")(fn)
        fn = click.option("--ignore-key", type=str, multiple=True, help="Keys to ignore")(fn)
        fn = click.option("--replace-key", type=str, multiple=True, help="Keys to replace, use `:` to separate old and new key parts")(fn)
        fn = click.option("--prefix", type=str, default=None, help="Prefix to add to keys")(fn)
        fn = click.option("--unsafe", is_flag=True, help="Allow loading unsafe checkpoints.", default=False)(fn)
        return fn
    return wrap

@click.group("checkpoint-tools")
def main() -> None:
    pass

@main.command("convert")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@precision_options()
@checkpoint_options()
def convert(
    input_file: str,
    name: Optional[str]=None,
    precision: str="full",
    overwrite: bool=False,
    ignore_key: List[str]=[],
    replace_key: List[str]=[],
    prefix: Optional[str]=None,
    unsafe: bool=False
) -> None:
    """
    Convert a PyTorch/SafeTensors checkpoint to SafeTensors format,
    optionally changing the precision of the (floating point) tensors.

    Writes the file with a suffix appropriate for the precision of the
    tensors in the state dictionary.
    """
    if name is None:
        name, _ = os.path.splitext(os.path.basename(input_file))

    replace_keys = dict(
        (key, value)
        for key, _, value in
        (key.partition(":") for key in replace_key)
    )

    metadata = load_metadata(input_file)
    state_dict = load_state_dict(input_file, unsafe=unsafe)
    state_dict = get_filtered_renamed_state_dict(
        state_dict,
        ignore_keys=ignore_key,
        replace_keys=replace_keys,
        prefix=prefix
    )
    convert_state_dict_dtype(state_dict, precision)
    extension = get_extension_for_state_dict(state_dict)
    output_file = f"{name}{extension}"

    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            click.echo(f"Output file {output_file} already exists, use --overwrite to replace")
            return

    click.echo(f"Writing {output_file}")
    safetensors.torch.save_file(state_dict, output_file, metadata=metadata)
    click.echo("Done!")

@main.command("convert-to-diffusers")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--name", type=str, default=None, help="Output file name")
@click.option("--model-type", type=str, default=None, help="Model type, default inferred from state dictionary")
@precision_options(include_quantization=True)
@checkpoint_options()
def convert_to_diffusers(
    input_file: str,
    name: Optional[str]=None,
    model_type: Optional[str]=None,
    precision: str="full",
    overwrite: bool=False,
    ignore_key: List[str]=[],
    replace_key: List[str]=[],
    prefix: Optional[str]=None,
    unsafe: bool=False
) -> None:
    """
    Convert a non-diffusers PyTorch/SafeTensors checkpoint to Diffusers
    format in SafeTensors.

    Writes the file with a suffix appropriate for the precision of the
    tensors in the state dictionary.

    Supported model types:

        Stable Diffusion 1.5

        Stable Diffusion XL

        Stable Diffusion 3.5

        FLUX.Dev

        FLUX.Schnell
    """
    if name is None:
        name, _ = os.path.splitext(os.path.basename(input_file))

    if precision in ["nf4", "int8"]:
        quantization = precision
    else:
        quantization = None

    replace_keys = dict(
        (key, value)
        for key, _, value in 
        (key.partition(":") for key in replace_key)
    )

    model_type, state_dicts = get_diffusers_state_dicts_from_checkpoint(
        input_file,
        model_type=model_type,
        unsafe=unsafe
    )

    for model_name, state_dict in state_dicts.items():
        state_dict = get_filtered_renamed_state_dict(
            state_dict,
            ignore_keys=ignore_key,
            replace_keys=replace_keys,
            prefix=prefix
        )
        convert_state_dict_dtype(state_dict, precision)
        if quantization is not None:
            state_dict = quantize_state_dict_for_model(
                state_dict,
                model_type=model_type,
                model_name=model_name,
                precision=quantization
            )

        extension = get_extension_for_state_dict(state_dict)
        output_file = f"{name}-{model_name}{extension}"

        if os.path.exists(output_file):
            if overwrite:
                os.remove(output_file)
            else:
                click.echo(f"Output file {output_file} already exists, use --overwrite to replace")
                continue

        click.echo(f"Writing {output_file}")
        safetensors.torch.save_file(state_dict, output_file)

    click.echo("Done!")

@main.command("combine")
@click.argument("input_files", type=click.Path(exists=True), nargs=-1)
@precision_options()
@checkpoint_options()
def combine(
    input_files: List[str],
    name: Optional[str]=None,
    precision: str="full",
    overwrite: bool=False,
    ignore_key: List[str]=[],
    replace_key: List[str]=[],
    prefix: Optional[str]=None,
    unsafe: bool=False
) -> None:
    """
    Combine multiple PyTorch/SafeTensors checkpoints into a single
    SafeTensors checkpoint, optionally changing the precision of
    the (floating point) tensors.
    """
    replace_keys = dict(
        (key, value)
        for key, _, value
        in (key.partition(":") for key in replace_key)
    )

    if name is None:
        name = "-".join([
            os.path.splitext(os.path.basename(input_file))[0]
            for input_file in input_files
        ])

    combined_state_dict = {}
    combined_metadata = {}
    for input_file in input_files:
        state_dict = load_state_dict(input_file, unsafe=unsafe)
        combined_state_dict.update(state_dict)

        metadata = load_metadata(input_file)
        combined_metadata.update(metadata)

    combined_state_dict = get_filtered_renamed_state_dict(
        combined_state_dict,
        ignore_keys=ignore_key,
        replace_keys=replace_keys,
        prefix=prefix
    )
    convert_state_dict_dtype(combined_state_dict, precision)
    extension = get_extension_for_state_dict(combined_state_dict)
    output_file = f"{name}{extension}"
    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            click.echo(f"Output file {output_file} already exists, use --overwrite to replace")
            return

    click.echo(f"Writing {output_file}")
    safetensors.torch.save_file(combined_state_dict, output_file, metadata=combined_metadata)
    click.echo("Done!")

@main.command("metadata")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
def metadata(input_file: str) -> None:
    """
    Print metadata of a SafeTensors checkpoint.
    """
    state_dict = load_state_dict(input_file)
    metadata = load_metadata(input_file)

    total_params = 0
    for key, value in state_dict.items():
        shape = ", ".join([str(s) for s in value.shape])
        dtype = f"{value.dtype}".split(".")[-1]
        click.echo(f"{cyan(key)}: [{green(shape)}] <{dtype}>")
        total_params += value.numel()

    abbreviated_params = float(total_params)
    abbreviated_units = ["", "K", "M", "B", "T"]
    for unit in abbreviated_units:
        if abbreviated_params < 1000:
            break
        abbreviated_params /= 1000
    precision = 0 if unit == "" else 1 if abbreviated_params < 10 else 0
    abbreviated = "{{0:.{0}f}}{{1}}".format(precision).format(abbreviated_params, unit)
    click.echo()
    click.echo(f"Total parameters: {cyan(abbreviated)} ({total_params:,d})")

    if metadata:
        click.echo()
        click.echo("Metadata:")

        for key, metadatum in metadata.items():
            click.echo(f"{cyan(key)}: {green(metadatum)}")

if __name__ == "__main__":
    main()
