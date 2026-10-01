#!/usr/bin/env bash

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/common.sh"
source "${script_dir}/bench-common.sh"

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'EOF'
Usage: bench-bagel.sh [MODE] [PROMPT] [INPUT_IMAGE]

Defaults: image, 256x256, 8 steps, CFG 4, 1 warmup, 3 measured runs.
Set REPEATS, WARMUP, BENCH_DIR or any run-bagel.sh environment override.
Each measured run is a fresh process and includes model loading.
EOF
    exit 0
fi

mode=${1:-image}
prompt=${2:-A red apple on a rustic wooden table, studio photograph, detailed}
input=${3:-${INPUT:-}}
export WIDTH=${WIDTH:-256} HEIGHT=${HEIGHT:-256} STEPS=${STEPS:-8} CFG=${CFG:-4}
ascend_validate_mode "${mode}"
ascend_benchmark bagel "${script_dir}/run-bagel.sh" "${mode}" "${prompt}" "${input}"
