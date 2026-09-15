#!/usr/bin/env bash

set -euo pipefail

ASCEND_BENCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

ascend_benchmark() {
    local model_name=$1
    local runner=$2
    local mode=$3
    local prompt=$4
    local input=$5

    local repeats=${REPEATS:-3}
    local warmup=${WARMUP:-1}
    local interval=${NPU_SAMPLE_INTERVAL:-1}
    [[ ${repeats} =~ ^[1-9][0-9]*$ ]] || { printf 'REPEATS must be positive\n' >&2; return 2; }
    [[ ${warmup} =~ ^[0-9]+$ ]] || { printf 'WARMUP must be non-negative\n' >&2; return 2; }
    [[ ${interval} =~ ^[1-9][0-9]*$ ]] || { printf 'NPU_SAMPLE_INTERVAL must be positive\n' >&2; return 2; }

    local stamp result_dir
    stamp=$(date +%Y%m%d-%H%M%S)
    result_dir=${BENCH_DIR:-${ASCEND_PROJECT_ROOT}/benchmark-results/${model_name}-${mode}-${stamp}}
    mkdir -p "${result_dir}/logs" "${result_dir}/outputs" "${result_dir}/npu"

    printf 'model,mode,phase,iteration,status,elapsed_seconds,output,log\n' >"${result_dir}/results.csv"
    {
        printf 'model=%s\nmode=%s\nprompt=%s\nrepeats=%s\nwarmup=%s\n' \
            "${model_name}" "${mode}" "${prompt}" "${repeats}" "${warmup}"
        env | LC_ALL=C sort | grep -E '^(MODEL_DIR|BUILD_DIR|DOCKER_IMAGE|DEVICES|STRICT_ACCEL|UNDERSTANDING_BACKEND|VISION_BACKEND|GENERATION_BACKEND|GENERATION_MAX_VRAM|WIDTH|HEIGHT|STEPS|CFG|IMAGE_CFG|SHIFT|SEED|MAX_TOKENS)=' || true
    } >"${result_dir}/configuration.txt"
    npu-smi info >"${result_dir}/npu-before.txt"

    local total=$((warmup + repeats)) index phase measured_index output log_file npu_file
    for ((index=1; index<=total; index++)); do
        if ((index <= warmup)); then
            phase=warmup
            measured_index=${index}
        else
            phase=measure
            measured_index=$((index - warmup))
        fi

        output="${result_dir}/outputs/${phase}-${measured_index}.png"
        log_file="${result_dir}/logs/${phase}-${measured_index}.log"
        npu_file="${result_dir}/npu/${phase}-${measured_index}.log"

        (
            while true; do
                date --iso-8601=ns
                npu-smi info
                sleep "${interval}"
            done
        ) >"${npu_file}" 2>&1 &
        local monitor_pid=$!

        local start_ns end_ns status elapsed
        start_ns=$(date +%s%N)
        set +e
        if [[ -n ${input} ]]; then
            OUTPUT="${output}" CONTAINER_NAME="umm-bench-${model_name}-${phase}-${measured_index}-$$" \
                "${runner}" "${mode}" "${prompt}" "${input}" >"${log_file}" 2>&1
        else
            OUTPUT="${output}" CONTAINER_NAME="umm-bench-${model_name}-${phase}-${measured_index}-$$" \
                "${runner}" "${mode}" "${prompt}" >"${log_file}" 2>&1
        fi
        status=$?
        set -e
        end_ns=$(date +%s%N)
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
        elapsed=$(awk -v start="${start_ns}" -v end="${end_ns}" 'BEGIN { printf "%.3f", (end-start)/1000000000 }')

        if ! ascend_mode_writes_image "${mode}"; then
            output=
        fi
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${model_name}" "${mode}" "${phase}" "${measured_index}" "${status}" \
            "${elapsed}" "${output}" "${log_file}" >>"${result_dir}/results.csv"
        printf '%s %s/%s: status=%s elapsed=%ss\n' \
            "${phase}" "${measured_index}" "$([[ ${phase} == warmup ]] && printf '%s' "${warmup}" || printf '%s' "${repeats}")" \
            "${status}" "${elapsed}"
        if ((status != 0)); then
            printf 'failed run log: %s\n' "${log_file}" >&2
            return "${status}"
        fi
    done

    npu-smi info >"${result_dir}/npu-after.txt"
    {
        printf 'Key engine timings extracted from logs\n'
        grep -H -E 'sampling completed|generate_image completed|decoded, taking|offloaded [0-9]+/[0-9]+ layers|graph-cut layer split' \
            "${result_dir}"/logs/*.log || true
    } >"${result_dir}/engine-timings.txt"
    awk -F, '
        $3 == "measure" && $5 == 0 {
            value = $6 + 0
            if (count == 0 || value < minimum) minimum = value
            if (count == 0 || value > maximum) maximum = value
            total += value
            count++
        }
        END {
            if (count > 0) {
                printf "successful_runs=%d\nmin_seconds=%.3f\nmean_seconds=%.3f\nmax_seconds=%.3f\n", count, minimum, total/count, maximum
            } else {
                print "successful_runs=0"
            }
        }
    ' "${result_dir}/results.csv" >"${result_dir}/summary.txt"

    printf 'benchmark results: %s\n' "${result_dir}"
    cat "${result_dir}/results.csv"
    cat "${result_dir}/summary.txt"
}
