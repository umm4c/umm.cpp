#!/usr/bin/env python3
"""Create a self-contained UMM model package from an official checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "third_party" / "llama.cpp"
sys.path.insert(0, str(LLAMA / "gguf-py"))

import gguf
import numpy as np
from safetensors import safe_open
import torch


@dataclass(frozen=True)
class ModelSpec:
    """Model-family facts needed by the conversion pipeline."""

    architecture: str
    shard_prefix: str
    required_generation_tensors: frozenset[str]
    extra_components: tuple[str, ...] = ()

    @property
    def is_bagel(self) -> bool:
        return self.architecture == "bagel"


U1_SPEC = ModelSpec(
    architecture="sensenova_u1",
    shard_prefix="model",
    required_generation_tensors=frozenset({
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
        "vision_model.embeddings.patch_embedding.weight",
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
        "fm_modules.fm_head.conv1.weight",
    }),
)

BAGEL_SPEC = ModelSpec(
    architecture="bagel",
    shard_prefix="ema",
    required_generation_tensors=frozenset({
        "language_model.model.layers.0.self_attn.q_proj_moe_gen.weight",
        "vae2llm.weight",
        "llm2vae.weight",
        "latent_pos_embed.pos_embed",
    }),
    extra_components=("vision", "vae"),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_bagel_config(config: dict) -> None:
    expected = {
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "num_hidden_layers": 28,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
    }
    llm = config["llm_config"]
    if any(llm.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported BAGEL language model dimensions")
    if not llm.get("qk_norm") or llm.get("rope_scaling") or llm.get("use_sliding_window"):
        raise ValueError("Unsupported BAGEL attention configuration")
    if config.get("latent_patch_size") != 2 or config["vae_config"].get("z_channels") != 16:
        raise ValueError("Unsupported BAGEL latent dimensions")


def validate_u1_config(config: dict) -> None:
    if config.get("architectures") != ["NEOChatModel"]:
        raise ValueError("Only dense SenseNova U1.5 checkpoints are supported")
    llm = config["llm_config"]
    expected = {
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 42,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000,
        "rope_theta_hw": 10000,
    }
    for key, value in expected.items():
        if llm.get(key) != value:
            raise ValueError(f"Unsupported U1.5 configuration: {key}={llm.get(key)!r}")
    if llm.get("num_experts", 0) or llm.get("attention_bias", False) or llm.get("rope_scaling"):
        raise ValueError("MoE, attention bias, and scaled RoPE are not supported")


def validate_config(source: Path) -> str:
    """Validate the official config and return the detected architecture."""

    config = read_json(source / "config.json")
    if config.get("architectures") == ["BagelForConditionalGeneration"]:
        validate_bagel_config(config)
        return BAGEL_SPEC.architecture
    validate_u1_config(config)
    return U1_SPEC.architecture


def spec_for(architecture: str) -> ModelSpec:
    if architecture == BAGEL_SPEC.architecture:
        return BAGEL_SPEC
    if architecture == U1_SPEC.architecture:
        return U1_SPEC
    raise ValueError(f"Unsupported architecture: {architecture}")


def source_shards(source: Path, architecture: str = "sensenova_u1") -> list[Path]:
    """Resolve and validate all safetensors shards for a checkpoint."""

    spec = spec_for(architecture)
    index = source / f"{spec.shard_prefix}.safetensors.index.json"
    names = (
        sorted(set(read_json(index)["weight_map"].values()))
        if index.exists()
        else [f"{spec.shard_prefix}.safetensors"]
    )
    shards = []
    for name in names:
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"Missing or external checkpoint shard: {name}")
        shards.append(path)
    return shards


def generation_tensor(name: str, spec: ModelSpec) -> bool:
    if spec.is_bagel:
        return (
            name.startswith("language_model.") and "_moe_gen" in name
        ) or name.startswith(("vae2llm.", "llm2vae.", "time_embedder.", "latent_pos_embed."))
    return (
        name.startswith("language_model.") and "_mot_gen" in name
    ) or name.startswith((
        "vision_model.",
        "fm_modules.vision_model_mot_gen.embeddings.",
        "fm_modules.timestep_embedder.",
        "fm_modules.noise_scale_embedder.",
        "fm_modules.fm_head.",
    ))


def is_generation(name: str, architecture: str = "sensenova_u1") -> bool:
    """Compatibility wrapper for callers of the former helper."""

    return generation_tensor(name, spec_for(architecture))


def tensor_type_info(dtype: str) -> tuple[np.dtype, gguf.GGMLQuantizationType]:
    types = {
        "BF16": (np.dtype("uint16"), gguf.GGMLQuantizationType.BF16),
        "F16": (np.dtype("float16"), gguf.GGMLQuantizationType.F16),
        "F32": (np.dtype("float32"), gguf.GGMLQuantizationType.F32),
    }
    try:
        return types[dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported generation tensor dtype: {dtype}") from error


def selected_generation_tensors(
    shards: Iterable[Path], spec: ModelSpec
) -> list[tuple[Path, list[str]]]:
    selected = []
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            names = sorted(name for name in tensors.keys() if generation_tensor(name, spec))
        selected.append((shard, names))
    return selected


def write_generation(
    shards: list[Path],
    output: Path,
    architecture: str = "sensenova_u1",
    outtype: str | None = None,
) -> int:
    """Write the generation-only GGUF while streaming source tensors."""

    spec = spec_for(architecture)
    selected = selected_generation_tensors(shards, spec)
    names = {name for _, shard_names in selected for name in shard_names}
    missing = spec.required_generation_tensors - names
    if missing:
        raise ValueError(f"Missing generation tensors: {sorted(missing)}")

    writer = gguf.GGUFWriter(output, architecture)
    try:
        for shard, shard_names in selected:
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                for name in shard_names:
                    tensor = tensors.get_slice(name)
                    source_type = tensor.get_dtype()
                    target_type = (
                        outtype.upper()
                        if source_type == "BF16" and outtype in ("f16", "f32")
                        else source_type
                    )
                    dtype, ggml_type = tensor_type_info(target_type)
                    shape = tensor.get_shape()
                    writer.add_tensor_info(
                        name,
                        shape,
                        dtype,
                        math.prod(shape) * dtype.itemsize,
                        raw_dtype=ggml_type,
                    )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        # Stream one tensor at a time, preserving source precision and bytes.
        for shard, shard_names in selected:
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                for name in shard_names:
                    tensor = tensors.get_tensor(name)
                    if tensor.dtype == torch.bfloat16 and outtype in ("f16", "f32"):
                        data = tensor.to(torch.float16 if outtype == "f16" else torch.float32)
                    else:
                        data = tensor.view(torch.uint16) if tensor.dtype == torch.bfloat16 else tensor
                    writer.write_tensor_data(data.numpy())
                    del tensor, data
    finally:
        writer.close()
    return len(names)


def run_converter(source: Path, output: Path, *extra_args: str) -> None:
    command = [
        sys.executable,
        str(LLAMA / "convert_hf_to_gguf.py"),
        str(source),
        *extra_args,
        "--outfile",
        str(output),
    ]
    subprocess.run(command, check=True)


def copy_bagel_components(source: Path, package: Path, manifest: dict) -> None:
    run_converter(source, package / "vision.gguf", "--mmproj", "--outtype", "f16")
    vae = (source / "ae.safetensors").resolve(strict=True)
    if not vae.is_relative_to(source):
        raise ValueError("External BAGEL VAE checkpoint")
    shutil.copyfile(vae, package / "vae.safetensors")
    manifest["components"].update(vision="vision.gguf", vae="vae.safetensors")


def build_manifest(spec: ModelSpec) -> dict:
    components = {
        "understanding": "understanding.gguf",
        "generation": "generation.gguf",
    }
    if "vision" in spec.extra_components:
        components["vision"] = "vision.gguf"
    if "vae" in spec.extra_components:
        components["vae"] = "vae.safetensors"
    return {
        "format": "umm",
        "version": 2,
        "architecture": spec.architecture,
        "components": components,
    }


def convert(source: Path, output: Path, outtype: str, generation_outtype: str | None = None) -> None:
    """Convert a checkpoint into a package using an atomic temporary directory."""

    source = source.resolve(strict=True)
    output = output.absolute()
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a new directory")

    spec = spec_for(validate_config(source))
    shards = source_shards(source, spec.architecture)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        package = Path(temporary) / "package"
        package.mkdir()
        run_converter(
            source,
            package / "understanding.gguf",
            "--outtype",
            outtype,
        )
        tensor_count = write_generation(
            shards,
            package / "generation.gguf",
            spec.architecture,
            generation_outtype,
        )
        manifest = build_manifest(spec)
        if spec.is_bagel:
            copy_bagel_components(source, package, manifest)
        (package / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
        package.rename(output)

    print(
        f"Created {output} ({tensor_count} generation tensors; "
        "tokenizer embedded in understanding.gguf)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Official dense U1.5 or BAGEL checkpoint directory",
    )
    parser.add_argument("--output", required=True, type=Path, help="New model package directory")
    parser.add_argument(
        "--outtype",
        choices=("bf16", "f16", "f32", "q8_0"),
        default="bf16",
        help="Understanding weights format (default: bf16); generation preserves source precision",
    )
    parser.add_argument(
        "--generation-outtype",
        choices=("bf16", "f16", "f32"),
        help="Convert BF16 generation weights to this format (default: preserve source precision)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        convert(args.checkpoint, args.output, args.outtype, args.generation_outtype)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"umm conversion: {error}")


if __name__ == "__main__":
    main()
