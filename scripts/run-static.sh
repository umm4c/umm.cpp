#!/usr/bin/env bash
set -euo pipefail
if (($# < 3 || $# > 4)); then
    printf 'Usage: MODEL_DIR=/path/to/package %s local_310p|remote_910b text|understand|image PROMPT [IMAGE]\n' "$0" >&2
    exit 2
fi
policy=$1
mode=$2
prompt=$3
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model=$(realpath -- "${MODEL_DIR:?Set MODEL_DIR to the model package}")
[[ -f "$model/model.json" ]] || { printf 'Model package is missing model.json\n' >&2; exit 2; }
output=${OUTPUT_DIR:-$PWD/outputs/static-$policy-$(date +%Y%m%dT%H%M%S)-$$}
mkdir -p "$output"
output=$(realpath "$output")
image=${DOCKER_IMAGE:-umm-cann:8.5.0-mvp}
container=(run --rm -i --security-opt seccomp=unconfined --ulimit core=0 --network host
    -v "$project:/workspace/umm:ro" -v "$model:/model:ro"
    -v "$output:/output" -w /workspace/umm)
options=(--model /model --execution-policy "$policy" --mode "$mode" --prompt "$prompt"
    --n-ctx "${N_CTX:-8192}" --n-batch "${N_BATCH:-4096}" --n-ubatch "${N_UBATCH:-4096}"
    --max-tokens "${MAX_TOKENS:-64}" --steps "${STEPS:-8}" --cfg "${CFG:-1}"
    --shift "${SHIFT:-3}" --seed "${SEED:-7}")
case "$policy" in
    local_310p)
        physical=${LOCAL_NPU:-0}
        [[ "$physical" =~ ^[0-9]+$ ]] || { printf 'LOCAL_NPU must be a physical device number\n' >&2; exit 2; }
        for device in "/dev/davinci$physical" /dev/davinci_manager /dev/hisi_hdc /dev/devmm_svm; do
            [[ -e "$device" ]] || { printf 'Missing device %s\n' "$device" >&2; exit 2; }
            container+=(--device "$device")
        done
        container+=(-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -e "UMM_PHYSICAL_DEVICE=$physical")
        build=build-static-310p
        ;;
    remote_910b)
        options+=(--rpc-endpoint "${RPC_ENDPOINT:-127.0.0.1:50052}")
        container+=(-e GGML_RPC_NO_RDMA=1)
        build=build-rpc-client
        ;;
    *) printf 'Unknown static policy\n' >&2; exit 2 ;;
esac
case "$mode" in
    text) ;;
    understand)
        [[ $# == 4 ]] || { printf 'understand requires an input image\n' >&2; exit 2; }
        input=$(realpath -- "$4")
        container+=(-v "$input:/input/image:ro")
        options+=(--input /input/image)
        ;;
    image)
        options+=(--width "${WIDTH:-1024}" --height "${HEIGHT:-1024}" --output /output/image.png)
        ;;
    *) printf 'Unsupported mode; use text, understand or image\n' >&2; exit 2 ;;
esac
[[ -x "$project/$build/bin/umm-cli" ]] || { printf 'Build %s first\n' "$build" >&2; exit 2; }
printf 'policy=%s output=%s\n' "$policy" "$output" >&2
docker "${container[@]}" "$image" bash -lc '
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}
    exec "$@"
' bash "$build/bin/umm-cli" "${options[@]}"
