import torch
import safetensors

from typing import Dict, List, Optional, Tuple, Any, Iterator

from .log_util import logger
from .dtype_util import get_torch_dtype

__all__ = [
    "load_metadata",
    "load_state_dict",
    "get_state_dict_dtype",
    "get_filtered_renamed_state_dict",
    "convert_state_dict_dtype",
    "get_extension_for_state_dict"
]

def flattened_state_dict(
    state_dict: Dict[str, Any],
    prefix: Optional[str]=None
) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Flattens the state dict.
    """
    if isinstance(state_dict, torch.Tensor):
        yield "" if not prefix else prefix, state_dict
    elif isinstance(state_dict, dict):
        for key, value in state_dict.items():
            if prefix is not None:
                key = f"{prefix}.{key}"
            if isinstance(value, dict):
                yield from flattened_state_dict(value, key)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    yield from flattened_state_dict(state_dict, f"{key}.{i}")
            elif isinstance(value, torch.Tensor):
                yield key, value
            else:
                logger.warning(f"Skipping key {key} with value {value}.")

def flatten_state_dict(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Flattens the state dict.
    """
    return dict(flattened_state_dict(state_dict))

def load_state_dict(
    input_file: str,
    unsafe: bool=False
) -> Dict[str, torch.Tensor]:
    """
    Loads a state dictionary from a file
    """
    if input_file.endswith(".safetensors"):
        state_dict = {}
        with safetensors.safe_open(input_file, framework="pt") as f: # type: ignore[no-untyped-call]
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        state_dict = torch.load(
            input_file,
            map_location="cpu",
            weights_only=not unsafe
        )
        state_dict = flatten_state_dict(state_dict)
    return state_dict

def get_state_dict_dtype(
    state_dict: Dict[str, torch.Tensor]
) -> Optional[torch.dtype]:
    """
    Gets the dtype of the state dict.

    Only looks at floating-point tensors.
    When mixed-precision is detected, returns None.
    When same-precision-but-different-dtype is detected, returns the more common dtype. For example, when both bfloat16 and float16 are detected, returns float16.
    """
    all_are_float_32 = True
    all_are_float_16 = True
    all_are_bfloat_16 = True
    all_are_half = True
    all_are_float8_e4m3fn = True
    all_are_float8_e4m3fn_uz = True
    all_are_float8_e5m2 = True
    all_are_float8_e5m2fn_uz = True
    all_are_float8 = True

    for key, value in state_dict.items():
        if value.is_floating_point():
            all_are_float_32 &= value.dtype is torch.float32
            all_are_float_16 &= value.dtype is torch.float16
            all_are_bfloat_16 &= value.dtype is torch.bfloat16
            all_are_half &= (value.dtype is torch.float16 or value.dtype is torch.bfloat16)
            all_are_float8_e4m3fn &= value.dtype is torch.float8_e4m3fn
            all_are_float8_e4m3fn_uz &= value.dtype is torch.float8_e4m3fnuz
            all_are_float8_e5m2 &= value.dtype is torch.float8_e5m2
            all_are_float8_e5m2fn_uz &= value.dtype is torch.float8_e5m2fnuz
            all_are_float8 &= (
                value.dtype is torch.float8_e4m3fn or \
                value.dtype is torch.float8_e5m2 or \
                value.dtype is torch.float8_e4m3fnuz or \
                value.dtype is torch.float8_e5m2fnuz
            )

    if all_are_float_32:
        return torch.float32
    elif all_are_bfloat_16:
        return torch.bfloat16
    elif all_are_float_16 or all_are_half:
        return torch.float16
    elif all_are_float8_e5m2:
        return torch.float8_e5m2
    elif all_are_float8_e5m2fn_uz:
        return torch.float8_e5m2fnuz
    elif all_are_float8_e4m3fn:
        return torch.float8_e4m3fn
    elif all_are_float8_e4m3fn_uz:
        return torch.float8_e4m3fnuz
    elif all_are_float8:
        return torch.float8_e4m3fn

    return None

def get_filtered_renamed_state_dict(
    state_dict: Dict[str, torch.Tensor],
    ignore_keys: List[str]=[],
    replace_keys: Dict[str, str]={},
    prefix: Optional[str]=None
) -> Dict[str, torch.Tensor]:
    """
    Filters the state dict by ignoring and renaming keys.
    """
    filtered_state_dict = {}
    for key, value in state_dict.items():
        for ignored_key in ignore_keys:
            if ignored_key in key:
                continue
        for old, new in replace_keys.items():
            if old in key:
                key = key.replace(old, new)
        if prefix is not None:
            if prefix.endswith("."):
                key = f"{prefix}{key}"
            else:
                key = f"{prefix}.{key}"
        if key in filtered_state_dict:
            raise ValueError(f"Key {key} already exists in the state dict.")
        filtered_state_dict[key] = value
    return filtered_state_dict

def convert_state_dict_dtype(
    state_dict: Dict[str, torch.Tensor],
    precision: str="full"
) -> None:
    """
    Converts the state dict to a specific precision in-place.
    """
    if precision == "full":
        return

    if precision in ["nf4", "int8"]:
        will_quantize = True
        dtype = torch.float16
    else:
        will_quantize = False
        dtype = get_torch_dtype(precision)

    for key, value in state_dict.items():
        if value.is_floating_point():
            if (
                not will_quantize or
                (will_quantize and torch.finfo(value.dtype).bits == 8)
            ):
                # 8-bit needs to be upcasted to 16-bit before quantization.
                state_dict[key] = value.to(dtype)

def get_extension_for_state_dict(
    state_dict: Dict[str, torch.Tensor],
    quantization: Optional[str]=None
) -> str:
    """
    Gets the extension for the state dict.
    """
    state_dict_dtype = get_state_dict_dtype(state_dict)
    if state_dict_dtype is torch.bfloat16:
        precision_ext = ".bf16"
    elif state_dict_dtype is torch.float16:
        precision_ext = ".fp16"
    elif state_dict_dtype is torch.float8_e5m2:
        precision_ext = ".fp8-e5m2"
    elif state_dict_dtype is torch.float8_e5m2fnuz:
        precision_ext = ".fp8-e5m2-fn-uz"
    elif state_dict_dtype is torch.float8_e4m3fn:
        precision_ext = ".fp8-e4m3-fn"
    elif state_dict_dtype is torch.float8_e4m3fnuz:
        precision_ext = ".fp8-e4m3-fn-uz"
    else:
        precision_ext = ""

    if quantization is None:
        quant_ext = ""
    else:
        quant_ext = f".{quantization}"

    return f"{precision_ext}{quant_ext}.safetensors"

def load_metadata(input_file: str) -> Dict[str, str]:
    """
    Loads the metadata from a safetensors file.

    :param input_file: The path to the safetensors file.
    :return: The metadata as a dictionary.
    """
    if not input_file.endswith(".safetensors"):
        return {}

    with safetensors.safe_open(input_file, framework="pt") as f: # type: ignore[no-untyped-call]
        metadata = f.metadata()

    if not isinstance(metadata, dict):
        return {}

    return metadata
