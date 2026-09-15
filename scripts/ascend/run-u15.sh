#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    cat <<'EOF'
Usage: run-u15.sh [MODE] [PROMPT]

Modes supported by the validated U1.5 binary: text, image, think-image

Examples:
  scripts/ascend/run-u15.sh text '用一句话介绍华为昇腾。'
  STEPS=8 scripts/ascend/run-u15.sh image 'A red apple on a wooden table'

Common overrides:
  MODEL_DIR, BUILD_DIR, DOCKER_IMAGE, DEVICES, OUTPUT, STRICT_ACCEL
  UNDERSTANDING_BACKEND, GENERATION_BACKEND, GENERATION_MAX_VRAM,
  WIDTH, HEIGHT, STEPS, CFG, SHIFT, SEED, MAX_TOKENS
EOF
}

[[ ${1:-} != --help && ${1:-} != -h ]] || { usage; exit 0; }

mode=${1:-image}
prompt=${2:-A red apple on a rustic wooden table, studio photograph, detailed}
input=${3:-${INPUT:-}}
ascend_validate_mode "${mode}"
case "${mode}" in
    text|image|think-image) ;;
    *) ascend_die "the validated U1.5 build supports text, image, and think-image only" ;;
esac

# F32 generation is the validated U1.5 image path on Ascend 310P.
model_dir=${MODEL_DIR:-${ASCEND_PROJECT_ROOT}/models/SenseNova-U1.5-8B-MoT-F32Gen-UMM}
# Keep U1.5 on its validated binary: the BAGEL build uses a newer strict CANN
# support check that rejects U1.5's offset RoPE during graph reservation.
export BUILD_DIR=${BUILD_DIR:-build-cann-rebase}
understanding_backend=${UNDERSTANDING_BACKEND:-CANN0}
generation_backend=${GENERATION_BACKEND:-CANN0&CANN1}
generation_max_vram=${GENERATION_MAX_VRAM:-CANN0=12,CANN1=38}
width=${WIDTH:-256}
height=${HEIGHT:-256}
steps=${STEPS:-8}
cfg=${CFG:-4}
shift_value=${SHIFT:-3}
seed=${SEED:-42}
max_tokens=${MAX_TOKENS:-64}
output=${OUTPUT:-${ASCEND_PROJECT_ROOT}/outputs/u15-${mode}-${width}x${height}-${steps}step.png}

input=
if ! ascend_mode_writes_image "${mode}"; then
    output=
fi

args=(--mode "${mode}" --prompt "${prompt}" --understanding-backend "${understanding_backend}")
case "${mode}" in
    text)
        args+=(--max-tokens "${max_tokens}")
        ;;
    *)
        args+=(--width "${width}" --height "${height}" --steps "${steps}"
               --cfg "${cfg}" --shift "${shift_value}"
               --seed "${seed}" --max-tokens "${max_tokens}"
               --generation-backend "${generation_backend}"
               --generation-max-vram "${generation_max_vram}")
        ;;
esac

export DEVICES=${DEVICES:-2,3}
ascend_run_umm "${model_dir}" "${input}" "${output}" "${args[@]}"
