#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    cat <<'EOF'
Usage: run-bagel.sh [MODE] [PROMPT] [INPUT_IMAGE]

Modes: text, image, think-image, understand, think-understand, edit, think-edit

Examples:
  scripts/ascend/run-bagel.sh text 'Describe Ascend NPU.'
  STEPS=8 scripts/ascend/run-bagel.sh image 'A red apple on a wooden table'
  scripts/ascend/run-bagel.sh understand 'What is shown?' outputs/apple.png

Common overrides:
  MODEL_DIR, BUILD_DIR, DOCKER_IMAGE, DEVICES, OUTPUT, STRICT_ACCEL
  UNDERSTANDING_BACKEND, VISION_BACKEND, GENERATION_BACKEND,
  GENERATION_MAX_VRAM, WIDTH, HEIGHT, STEPS, CFG, IMAGE_CFG, SHIFT,
  SEED, MAX_TOKENS, VAE_TILING, VAE_TILE_SIZE, VAE_TILE_OVERLAP
EOF
}

[[ ${1:-} != --help && ${1:-} != -h ]] || { usage; exit 0; }

mode=${1:-image}
prompt=${2:-A red apple on a rustic wooden table, studio photograph, detailed}
input=${3:-${INPUT:-}}
ascend_validate_mode "${mode}"

model_dir=${MODEL_DIR:-${ASCEND_PROJECT_ROOT}/models/BAGEL-7B-MoT-F16-UMM}
understanding_backend=${UNDERSTANDING_BACKEND:-CANN0}
vision_backend=${VISION_BACKEND:-CANN1}
generation_backend=${GENERATION_BACKEND:-diffusion=CANN1&CANN2,vae=CANN0}
generation_max_vram=${GENERATION_MAX_VRAM:-CANN1=14,CANN2=14}
width=${WIDTH:-256}
height=${HEIGHT:-256}
steps=${STEPS:-8}
cfg=${CFG:-4}
image_cfg=${IMAGE_CFG:-1.5}
shift_value=${SHIFT:-3}
seed=${SEED:-42}
max_tokens=${MAX_TOKENS:-64}
vae_tiling=${VAE_TILING:-$((width >= 1024 || height >= 1024 ? 1 : 0))}
vae_tile_size=${VAE_TILE_SIZE:-64}
vae_tile_overlap=${VAE_TILE_OVERLAP:-0.5}
output=${OUTPUT:-${ASCEND_PROJECT_ROOT}/outputs/bagel-${mode}-${width}x${height}-${steps}step.png}

if ascend_mode_needs_input "${mode}"; then
    [[ -n ${input} ]] || ascend_die "mode '${mode}' requires INPUT_IMAGE or INPUT"
else
    input=
fi
if ! ascend_mode_writes_image "${mode}"; then
    output=
fi

args=(--mode "${mode}" --prompt "${prompt}" --understanding-backend "${understanding_backend}")
case "${mode}" in
    text)
        args+=(--max-tokens "${max_tokens}")
        ;;
    understand|think-understand)
        args+=(--max-tokens "${max_tokens}" --vision-backend "${vision_backend}")
        ;;
    *)
        args+=(--width "${width}" --height "${height}" --steps "${steps}"
               --cfg "${cfg}" --image-cfg "${image_cfg}" --shift "${shift_value}"
               --seed "${seed}" --max-tokens "${max_tokens}"
               --vae-tiling "${vae_tiling}" --vae-tile-size "${vae_tile_size}"
               --vae-tile-overlap "${vae_tile_overlap}"
               --vision-backend "${vision_backend}"
               --generation-backend "${generation_backend}"
               --generation-max-vram "${generation_max_vram}")
        ;;
esac

export DEVICES=${DEVICES:-2,3,4}
ascend_run_umm "${model_dir}" "${input}" "${output}" "${args[@]}"
