import torch
import numpy as np

from typing import Any, Type, Union

__all__ = ["get_torch_dtype"]

def get_torch_dtype_from_string(torch_type: str) -> torch.dtype:
    """
    Converts a string to a torch DType.
    """
    return {
        "complex128": torch.complex128,
        "cdouble": torch.complex128,
        "complex": torch.complex64,
        "complex64": torch.complex64,
        "cfloat": torch.complex64,
        "cfloat64": torch.complex64,
        "cf64": torch.complex64,
        "double": torch.float64,
        "float64": torch.float64,
        "fp64": torch.float64,
        "float": torch.float32,
        "full": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
        "float8": torch.float8_e4m3fn,
        "float8_e4m3": torch.float8_e4m3fn,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e4m3_fn": torch.float8_e4m3fn,
        "float8_e4m3fnuz": torch.float8_e4m3fnuz,
        "float8_e4m3_fnuz": torch.float8_e4m3fnuz,
        "float8_e4m3_fn_uz": torch.float8_e4m3fnuz,
        "float8_e5m2": torch.float8_e5m2,
        "float8_e5m2fnuz": torch.float8_e5m2fnuz,
        "float8_e5m2_fnuz": torch.float8_e5m2fnuz,
        "float8_e5m2_fn_uz": torch.float8_e5m2fnuz,
        "fp85": torch.float8_e5m2,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "short": torch.int16,
        "int": torch.int32,
        "int32": torch.int32,
        "long": torch.int64,
        "int64": torch.int64,
        "bool": torch.bool,
        "bit": torch.bool,
        "1": torch.bool
    }[
        (torch_type[6:] if torch_type.startswith("torch.") else torch_type)
            .lower()
            .replace("-", "_")
            .strip()
    ]

def get_torch_dtype_from_numpy_dtype(numpy_type: Type[Any]) -> torch.dtype:
    """
    Gets the torch type from a numpy type.
    :raises: KeyError When type is unknown.
    """
    return {
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
        np.bool_: torch.bool,
    }[numpy_type]

def get_torch_dtype(
    type_to_convert: Union[torch.dtype, Type[Any], str]
) -> torch.dtype:
    """
    Gets the torch data type from a string, numpy type, or torch type.
    """
    if isinstance(type_to_convert, torch.dtype):
        return type_to_convert
    if isinstance(type_to_convert, str):
        return get_torch_dtype_from_string(type_to_convert) # Raises

    try:
        return get_torch_dtype_from_numpy_dtype(type_to_convert)
    except KeyError:
        return get_torch_dtype_from_string(str(type_to_convert)) # Raises
