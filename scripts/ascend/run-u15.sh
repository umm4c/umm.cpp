#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    cat <<'EOF'
Usage: run-u15.sh [MODE] [PROMPT]

Modes: text, image, think-image, interleave, think-interleave

Examples:
  scripts/ascend/run-u15.sh text '用一句话介绍华为昇腾。'
  scripts/ascend/run-u15.sh image 'A red apple on a wooden table'

Common overrides:
  MODEL_DIR, BUILD_DIR, DOCKER_IMAGE, DEVICES, OUTPUT, STRICT_ACCEL
  UNDERSTANDING_BACKEND, GENERATION_BACKEND, GENERATION_MAX_VRAM,
  WIDTH, HEIGHT, STEPS, CFG, SHIFT, SEED, N_CTX, N_BATCH, N_UBATCH,
  MAX_TOKENS, MAX_IMAGES
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

# Default to the validated single-card U1.5 8-step LoRA V2 package.
model_dir=${MODEL_DIR:-${ASCEND_PROJECT_ROOT}/models/SenseNova-U1.5-8B-MoT-LoRA-8step-V2-Q8U-F16G-UMM}
understanding_backend=${UNDERSTANDING_BACKEND:-CANN0}
generation_backend=${GENERATION_BACKEND:-CANN0}
generation_max_vram=${GENERATION_MAX_VRAM:-CANN0=40}
if [[ ${mode} == interleave || ${mode} == think-interleave ]]; then
    default_size=512
else
    default_size=1024
fi
width=${WIDTH:-${default_size}}
height=${HEIGHT:-${default_size}}
steps=${STEPS:-8}
cfg=${CFG:-1}
shift_value=${SHIFT:-3}
seed=${SEED:-42}
max_tokens=${MAX_TOKENS:-64}
max_images=${MAX_IMAGES:-2}
n_ctx=${N_CTX:-2048}
n_batch=${N_BATCH:-512}
n_ubatch=${N_UBATCH:-256}
if [[ ${mode} == interleave || ${mode} == think-interleave ]]; then
    output=${OUTPUT:-${ASCEND_PROJECT_ROOT}/outputs/u15-v2-${mode}-${width}x${height}-${steps}step}
else
    output=${OUTPUT:-${ASCEND_PROJECT_ROOT}/outputs/u15-v2-${mode}-${width}x${height}-${steps}step.png}
fi

input=
if ! ascend_mode_writes_image "${mode}"; then
    output=
fi

args=(--mode "${mode}" --prompt "${prompt}" --understanding-backend "${understanding_backend}"
      --n-ctx "${n_ctx}" --n-batch "${n_batch}" --n-ubatch "${n_ubatch}")
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
export BUILD_DIR=${BUILD_DIR:-build-cann-rebase}
ascend_run_umm "${model_dir}" "${input}" "${output}" "${args[@]}"
