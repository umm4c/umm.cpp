#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    cat <<'EOF'
Usage: run-u15.sh [MODE] [PROMPT]

Modes: text, image, think-image, interleave, think-interleave

Examples:
  scripts/ascend/run-u15.sh text '用一句话介绍华为昇腾。'
  STEPS=8 scripts/ascend/run-u15.sh image 'A red apple on a wooden table'

Common overrides:
  MODEL_DIR, BUILD_DIR, DOCKER_IMAGE, DEVICES, OUTPUT, STRICT_ACCEL
  UNDERSTANDING_BACKEND, GENERATION_BACKEND, GENERATION_MAX_VRAM,
  WIDTH, HEIGHT, STEPS, CFG, SHIFT, SEED, MAX_TOKENS, MAX_IMAGES
EOF
}

[[ ${1:-} != --help && ${1:-} != -h ]] || { usage; exit 0; }

mode=${1:-image}
prompt=${2:-A red apple on a rustic wooden table, studio photograph, detailed}
input=${3:-${INPUT:-}}
ascend_validate_mode "${mode}"
case "${mode}" in
    text|image|think-image|interleave|think-interleave) ;;
    *) ascend_die "U1.5 does not support mode '${mode}'" ;;
esac

# The validated single-card U1.5 path keeps Q8 understanding and F16 generation resident.
model_dir=${MODEL_DIR:-${ASCEND_PROJECT_ROOT}/models/SenseNova-U1.5-8B-MoT-Q8U-F16G-UMM}
understanding_backend=${UNDERSTANDING_BACKEND:-CANN0}
generation_backend=${GENERATION_BACKEND:-CANN0}
generation_max_vram=${GENERATION_MAX_VRAM:-CANN0=40}
width=${WIDTH:-256}
height=${HEIGHT:-256}
steps=${STEPS:-8}
if [[ ${mode} == interleave || ${mode} == think-interleave ]]; then
    cfg=${CFG:-1}
else
    cfg=${CFG:-4}
fi
shift_value=${SHIFT:-3}
seed=${SEED:-42}
max_tokens=${MAX_TOKENS:-64}
max_images=${MAX_IMAGES:-2}
if [[ ${mode} == interleave || ${mode} == think-interleave ]]; then
    output=${OUTPUT:-${ASCEND_PROJECT_ROOT}/outputs/u15-${mode}-${width}x${height}-${steps}step}
else
    output=${OUTPUT:-${ASCEND_PROJECT_ROOT}/outputs/u15-${mode}-${width}x${height}-${steps}step.png}
fi

input=
if ! ascend_mode_writes_image "${mode}"; then
    output=
fi

args=(--mode "${mode}" --prompt "${prompt}" --understanding-backend "${understanding_backend}")
case "${mode}" in
    text)
        args+=(--max-tokens "${max_tokens}")
        ;;
    interleave|think-interleave)
        args+=(--width "${width}" --height "${height}" --steps "${steps}"
               --cfg "${cfg}" --shift "${shift_value}"
               --seed "${seed}" --max-tokens "${max_tokens}" --max-images "${max_images}"
               --generation-backend "${generation_backend}"
               --generation-max-vram "${generation_max_vram}")
        ;;
    *)
        args+=(--width "${width}" --height "${height}" --steps "${steps}"
               --cfg "${cfg}" --shift "${shift_value}"
               --seed "${seed}" --max-tokens "${max_tokens}"
               --generation-backend "${generation_backend}"
               --generation-max-vram "${generation_max_vram}")
        ;;
esac

export DEVICES=${DEVICES:-2}
ascend_run_umm "${model_dir}" "${input}" "${output}" "${args[@]}"
