#!/usr/bin/env bash

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/common.sh"
source "${script_dir}/bench-common.sh"

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'EOF'
Usage: bench-u15.sh [MODE] [PROMPT]

Defaults: U1.5 LoRA V2 image, 1024x1024, 8 steps, CFG 1, 1 warmup, 3 measured runs.
Set REPEATS, WARMUP, BENCH_DIR or any run-u15.sh environment override.
Each measured run is a fresh process and includes model loading.
EOF
    exit 0
fi

mode=${1:-image}
prompt=${2:-A red apple on a rustic wooden table, studio photograph, detailed}
input=${3:-${INPUT:-}}
if [[ ${mode} == interleave || ${mode} == think-interleave ]]; then
    export WIDTH=${WIDTH:-512} HEIGHT=${HEIGHT:-512} STEPS=${STEPS:-8} CFG=${CFG:-1}
else
    export WIDTH=${WIDTH:-1024} HEIGHT=${HEIGHT:-1024} STEPS=${STEPS:-8} CFG=${CFG:-1}
fi
export U1_PREFILL_MODE=${U1_PREFILL_MODE:-mixed4}
export U1_V_CACHE_LAYOUT=${U1_V_CACHE_LAYOUT:-contiguous}
export CANN_Q8_NZ=${CANN_Q8_NZ:-1}
export N_CTX=${N_CTX:-2048} N_BATCH=${N_BATCH:-512} N_UBATCH=${N_UBATCH:-256}
ascend_validate_mode "${mode}"
ascend_benchmark u15 "${script_dir}/run-u15.sh" "${mode}" "${prompt}" "${input}"
