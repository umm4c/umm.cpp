#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
bundle=${root}/artifacts/runtime/cann310p-20261020
version=umm-cann:8.5.0-310p-runtime-20261020
guard=umm-cann-runtime-keep
(cd "${bundle}" && sha256sum -c umm-cann-runtime-20261020.tar.gz.sha256)
docker load -i "${bundle}/umm-cann-runtime-20261020.tar.gz"
expected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' "${root}/docker/runtime-cann310p.lock.json")
image_id=$(docker image inspect --format '{{.Id}}' "${version}")
[[ ${image_id} == "${expected}" ]] || { printf 'Unexpected restored image ID.\n' >&2; exit 1; }
docker tag "${version}" umm-cann:8.5.0-mvp
if docker container inspect "${guard}" >/dev/null 2>&1; then
    [[ $(docker container inspect --format '{{.Image}}' "${guard}") == "${image_id}" ]] || {
        printf 'Protection container uses another image; retained unchanged.\n' >&2; exit 1;
    }
else
    docker create --name "${guard}" --label umm.environment.keep=true --entrypoint /bin/true "${version}"
fi
printf 'Runtime restored: %s\n' "${image_id}"
