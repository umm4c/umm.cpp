#!/usr/bin/env bash
# Run on the worker host. Config is a trusted shell environment file.
set -euo pipefail
usage() { echo "Usage: $0 build|start|stop|status|logs CONFIG" >&2; exit 2; }
[[ $# == 2 ]] || usage
action=$1
case "$action" in build|start|stop|status|logs) ;; *) usage ;; esac
source "$2"
: "${RPC_WORKDIR:?}" "${RPC_IMAGE:?}" "${RPC_EXPECTED_IMAGE_ID:?}"
: "${RPC_CONTAINER:=umm-rpc-910b-worker}" "${RPC_NPU:=4}" "${RPC_PORT:=50052}"
: "${RPC_SOC:=Ascend910B1}" "${RPC_BUILD_JOBS:=4}"
[[ $RPC_WORKDIR == /* && $RPC_NPU =~ ^[0-9]+$ && $RPC_PORT =~ ^[0-9]+$ && $RPC_BUILD_JOBS =~ ^[1-9][0-9]*$ ]] || usage
if [[ $action == stop ]]; then docker stop "$RPC_CONTAINER"; exit; fi
if [[ $action == status ]]; then docker inspect "$RPC_CONTAINER"; exit; fi
if [[ $action == logs ]]; then docker logs "$RPC_CONTAINER"; exit; fi
actual=$(docker image inspect --format '{{.Id}}' "$RPC_IMAGE")
[[ $actual == "$RPC_EXPECTED_IMAGE_ID" ]] || { echo "Image ID differs from configured environment" >&2; exit 1; }
common=(--network host --security-opt seccomp=unconfined --ulimit core=0
    -v "$RPC_WORKDIR:/work" -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
    -w /work --entrypoint bash)
if [[ $action == build ]]; then
    [[ -f $RPC_WORKDIR/llama.cpp/CMakeLists.txt ]] || { echo "Sync the pinned llama.cpp source first" >&2; exit 1; }
    mkdir -p "$RPC_WORKDIR/logs"
    # pipefail propagates build errors through tee; CANN environment setup runs before nounset.
    docker run --rm "${common[@]}" -e RPC_SOC="$RPC_SOC" -e RPC_BUILD_JOBS="$RPC_BUILD_JOBS" "$actual" -c '
        source /usr/local/Ascend/ascend-toolkit/set_env.sh || exit
        set -euo pipefail
        cmake -S /work/llama.cpp -B /work/build-rpc-910b \
          -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
          -DGGML_RPC=ON -DGGML_RPC_RDMA=OFF -DGGML_CANN=ON \
          -DSOC_TYPE="$RPC_SOC" -DUSE_ACL_GRAPH=OFF \
          -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=OFF \
          -DCMAKE_C_FLAGS=-DGGML_MAX_NAME=160 -DCMAKE_CXX_FLAGS=-DGGML_MAX_NAME=160
        cmake --build /work/build-rpc-910b --target ggml-rpc-server -j "$RPC_BUILD_JOBS"
        /work/build-rpc-910b/bin/ggml-rpc-server --help
    ' 2>&1 | tee "$RPC_WORKDIR/logs/build-worker.log"
    exit
fi
[[ -x $RPC_WORKDIR/build-rpc-910b/bin/ggml-rpc-server ]] || { echo "Build the worker first" >&2; exit 1; }
# Refuse a name collision: replacement requires an explicit stop and docker rm.
docker run -d --name "$RPC_CONTAINER" --label umm.rpc.worker=true \
    "${common[@]}" --restart no \
    --device "/dev/davinci$RPC_NPU" --device /dev/davinci_manager \
    --device /dev/hisi_hdc --device /dev/devmm_svm \
    -e GGML_RPC_NO_RDMA=1 -e GGML_CANN_Q8_NZ=0 -e GGML_CANN_SATURATION=1 \
    -e LLAMA_CACHE=/work/cache -e OMP_NUM_THREADS=1 -e RPC_PORT="$RPC_PORT" \
    "$actual" -c '
        source /usr/local/Ascend/ascend-toolkit/set_env.sh || exit
        set -euo pipefail
        export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}
        exec /work/build-rpc-910b/bin/ggml-rpc-server --host 127.0.0.1 --port "$RPC_PORT" --device CANN0 --cache
    '
