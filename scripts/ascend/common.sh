#!/usr/bin/env bash

set -euo pipefail

ASCEND_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ASCEND_PROJECT_ROOT=$(cd -- "${ASCEND_SCRIPT_DIR}/../.." && pwd)

ascend_die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

ascend_require_command() {
    command -v "$1" >/dev/null 2>&1 || ascend_die "required command not found: $1"
}

ascend_validate_mode() {
    case "$1" in
        text|image|think-image|understand|think-understand|edit|think-edit) ;;
        *) ascend_die "unsupported mode '$1'" ;;
    esac
}

ascend_mode_needs_input() {
    case "$1" in
        understand|think-understand|edit|think-edit) return 0 ;;
        *) return 1 ;;
    esac
}

ascend_mode_writes_image() {
    case "$1" in
        image|think-image|edit|think-edit) return 0 ;;
        *) return 1 ;;
    esac
}

ascend_device_args() {
    local devices_csv=$1
    local -n result_ref=$2
    local device

    IFS=',' read -r -a devices <<<"${devices_csv}"
    ((${#devices[@]} > 0)) || ascend_die 'DEVICES must contain at least one physical device id'
    for device in "${devices[@]}"; do
        [[ ${device} =~ ^[0-9]+$ ]] || ascend_die "invalid device id '${device}' in DEVICES"
        [[ -e /dev/davinci${device} ]] || ascend_die "/dev/davinci${device} does not exist"
        result_ref+=(--device "/dev/davinci${device}")
    done
    for device in /dev/davinci_manager /dev/hisi_hdc /dev/devmm_svm; do
        [[ -e ${device} ]] || ascend_die "${device} does not exist"
        result_ref+=(--device "${device}")
    done
}

# Run umm-cli in the pinned CANN container. Arguments:
#   1: host model directory
#   2: optional host input image
#   3: optional host output image
#   4+: umm-cli arguments after --model
ascend_run_umm() {
    local model_host=$1
    local input_host=$2
    local output_host=$3
    shift 3

    ascend_require_command docker
    [[ -d ${model_host} ]] || ascend_die "model directory not found: ${model_host}"

    local docker_image=${DOCKER_IMAGE:-umm-cann:8.5.0-mvp}
    local build_dir=${BUILD_DIR:-build-cann-bagel}
    local devices_csv=${DEVICES:-2,3,4}
    local container_name=${CONTAINER_NAME:-umm-ascend-$(date +%Y%m%d-%H%M%S)-$$}
    local strict_accel=${STRICT_ACCEL:-1}
    local output_container=
    local input_container=
    local -a docker_args=(
        run --rm --name "${container_name}"
        --security-opt seccomp=unconfined
        --network host
        -e "GGML_SCHED_STRICT_ACCEL=${strict_accel}"
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
        -v "${ASCEND_PROJECT_ROOT}:/workspace/umm:ro"
        -v "${model_host}:/model:ro"
    )

    ascend_device_args "${devices_csv}" docker_args

    if [[ -n ${input_host} ]]; then
        [[ -f ${input_host} ]] || ascend_die "input image not found: ${input_host}"
        input_host=$(realpath "${input_host}")
        input_container=/input/source-image
        docker_args+=(-v "${input_host}:${input_container}:ro")
    fi

    if [[ -n ${output_host} ]]; then
        local output_parent output_name
        output_parent=$(dirname -- "${output_host}")
        output_name=$(basename -- "${output_host}")
        mkdir -p "${output_parent}"
        output_parent=$(realpath "${output_parent}")
        output_container="/output/${output_name}"
        docker_args+=(-v "${output_parent}:/output" -e "UMM_OUTPUT_FILE=${output_container}")
    fi

    local -a cli_args=(--model /model "$@")
    [[ -z ${input_container} ]] || cli_args+=(--input "${input_container}")
    [[ -z ${output_container} ]] || cli_args+=(--output "${output_container}")

    docker "${docker_args[@]}" "${docker_image}" bash -lc '
        build_root=$1
        shift
        export LD_LIBRARY_PATH="${build_root}/bin:${LD_LIBRARY_PATH:-}"
        "${build_root}/bin/umm-cli" "$@"
        status=$?
        if [[ -n ${UMM_OUTPUT_FILE:-} ]]; then
            chmod a+r "${UMM_OUTPUT_FILE}" "${UMM_OUTPUT_FILE}.json" 2>/dev/null || true
        fi
        exit "${status}"
    ' bash "/workspace/umm/${build_dir}" "${cli_args[@]}"
}
