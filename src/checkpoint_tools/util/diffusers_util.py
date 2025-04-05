import os
import gc
import json
import torch
import requests
import tempfile
import safetensors

from typing import Any, Dict, Optional, Tuple

from .log_util import logger
from .dummy_util import DummyModel
from .state_dict_util import load_state_dict

V1_CONFIG = "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/raw/main/unet/config.json"
XL_CONFIG = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/unet/config.json"
V3_5_MEDIUM_CONFIG = "https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/raw/main/transformer/config.json"
V3_5_LARGE_CONFIG = "https://huggingface.co/stabilityai/stable-diffusion-3.5-large/raw/main/transformer/config.json"
FLUX_DEV_CONFIG = "https://huggingface.co/black-forest-labs/FLUX.1-dev/raw/main/transformer/config.json"
FLUX_SCHNELL_CONFIG = "https://huggingface.co/black-forest-labs/FLUX.1-schnell/raw/main/transformer/config.json"

def get_diffusers_config_url(model_type: str) -> str:
    """
    Gets the configuration URL for a model type
    """
    if model_type == "v1":
        config_url = V1_CONFIG
    elif model_type == "xl":
        config_url = XL_CONFIG
    elif model_type == "flux-dev":
        config_url = FLUX_DEV_CONFIG
    elif model_type == "flux-schnell":
        config_url = FLUX_SCHNELL_CONFIG
    elif model_type == "sd35_medium":
        config_url = V3_5_MEDIUM_CONFIG
    elif model_type == "sd35_large":
        config_url = V3_5_LARGE_CONFIG
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return config_url

def get_diffusers_config(model_type: str) -> Dict[str, Any]:
    """
    Gets the configuration for a model type
    """
    config_url = get_diffusers_config_url(model_type)
    headers = {}
    hf_token = os.getenv("HF_TOKEN", None)
    if hf_token is not None:
        headers["Authorization"] = f"Bearer {hf_token}"

    response = requests.get(config_url, headers=headers)
    response.raise_for_status()
    result = response.json()
    assert isinstance(result, dict), f"Expected a dictionary, got {type(result)}"
    return result

def quantize_state_dict_for_model(
    state_dict: Dict[str, torch.Tensor],
    model_type: str,
    model_name: str,
    precision: str
) -> Dict[str, torch.Tensor]:
    """
    Quantizes a supported model using BitsAndBytes.
    """
    try:
        import bitsandbytes # type: ignore[import-untyped]
        bitsandbytes # silence import warning
    except ImportError:
        raise ImportError("BitsAndBytes is not installed. Run `pip install bitsandbytes` to use quantization.")
    try:
        if model_type not in ["flux-schnell", "flux-dev", "sd35_large", "sd35_medium"]:
            raise ValueError(f"Unsupported model type for quantization: {model_type}")
        if model_name not in ["transformer"]:
            raise ValueError(f"Unsupported model for quantization: {model_type}.{model_name}")

        from diffusers.quantizers.quantization_config import BitsAndBytesConfig
        if model_type == "flux-dev" or model_type == "flux-schnell":
            from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel as TransformerModel
        elif model_type in ["sd35_large", "sd35_medium"]:
            from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel as TransformerModel
    except ImportError:
        raise ImportError("Diffusers is not installed or needs to be updated. Run `pip install -U diffusers`")

    # Create a temporary directory to hold the safetensors file
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Writing temporary files to {temp_dir}")
        model_path = os.path.join(temp_dir, "diffusion_pytorch_model.safetensors")
        safetensors.torch.save_file(state_dict, model_path)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        # Copy config.json to the temporary directory
        config_path = os.path.join(temp_dir, "config.json")
        config = get_diffusers_config(model_type)
        with open(config_path, "w") as f:
            f.write(json.dumps(config))

        # Load the model with the quantization config
        quant_config = None
        if precision == "nf4":
            quant_config = BitsAndBytesConfig(load_in_4bit=True) # type: ignore[no-untyped-call]
        elif precision == "int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True) # type: ignore[no-untyped-call]
        else:
            raise ValueError(f"Invalid precision for quantization: {precision}")

        transformer = TransformerModel.from_pretrained(
            temp_dir,
            quantization_config=quant_config
        )
        return transformer.state_dict() # type: ignore[no-any-return]

def get_diffusers_state_dicts_from_checkpoint(
    input_file: str,
    model_type: Optional[str]=None,
    unsafe: bool=False,
) -> Tuple[str, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Converts a supported diffusion model to a SafeTensors file.
    """
    try:
        from diffusers.loaders.single_file_utils import (
            infer_diffusers_model_type,
            convert_ldm_unet_checkpoint,
            convert_ldm_clip_checkpoint,
            convert_open_clip_checkpoint,
            convert_flux_transformer_checkpoint_to_diffusers,
            convert_sd3_transformer_checkpoint_to_diffusers,
        )
    except ImportError:
        raise ImportError("Diffusers is not installed. Run `pip install -U diffusers`")

    state_dict = load_state_dict(input_file, unsafe=unsafe)
    if model_type is None:
        model_type = infer_diffusers_model_type(state_dict) # type: ignore[no-untyped-call]

    has_v1_text_encoder = any(key.startswith("cond_stage_model.transformer.text_model") for key in state_dict.keys())
    has_xl_text_encoder = any(key.startswith("conditioner.embedders.0.transformer") for key in state_dict.keys())
    has_xl_text_encoder_2 = any(key.startswith("conditioner.embedders.1.model") for key in state_dict.keys())

    if model_type not in ["v1", "xl_base", "flux-dev", "flux-schnell", "sd35_large", "sd35_medium"]:
        raise ValueError(f"Unsupported model type: {model_type}")

    state_dicts = {}

    if model_type in ["flux-dev", "flux-schnell"]:
        logger.info(f"Converting {model_type} transformer")
        state_dicts["transformer"] = convert_flux_transformer_checkpoint_to_diffusers(state_dict) # type: ignore[no-untyped-call]
    elif model_type in ["sd35_large", "sd35_medium"]:
        logger.info(f"Converting {model_type} transformer")
        state_dicts["transformer"] = convert_sd3_transformer_checkpoint_to_diffusers(state_dict) # type: ignore[no-untyped-call]
    else:
        unet_config = get_diffusers_config(model_type.split("_")[0])
        logger.info(f"Converting {model_type} unet")
        state_dicts["unet"] = convert_ldm_unet_checkpoint(state_dict, unet_config) # type: ignore[no-untyped-call]

    if model_type == "v1":
        if has_v1_text_encoder:
            logger.info("Converting text-encoder")
            state_dicts["text-encoder"] = convert_ldm_clip_checkpoint(state_dict) # type: ignore[no-untyped-call]
    elif model_type == "xl_base":
        if has_xl_text_encoder:
            logger.info("Converting text-encoder")
            state_dicts["text-encoder"] = convert_ldm_clip_checkpoint(state_dict) # type: ignore[no-untyped-call]

        if has_xl_text_encoder_2:
            logger.info("Converting text-encoder-2")
            state_dicts["text-encoder-2"] = convert_open_clip_checkpoint( # type: ignore[no-untyped-call]
                DummyModel(projection_dim=1280),
                state_dict,
                prefix="conditioner.embedders.1.model."
            )

    return model_type, state_dicts
