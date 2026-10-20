#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
bundle=${root}/artifacts/runtime/cann310p-20261020
base=umm-cann-base:8.5.0-310p-20261020
expected=sha256:7c5790b80ec63ac20e91d7f587a3cfb42cc0d282bc6c8f5dcc4d654d25fa0068
version=umm-cann:8.5.0-310p-runtime-20261020
guard=umm-cann-runtime-keep
if (($# > 1)) || [[ ${1:-} != '' && ${1:-} != --download ]]; then
    printf 'usage: %s [--download]\n' "$0" >&2; exit 2
fi
if ! docker image inspect "${base}" >/dev/null 2>&1; then
    source=quay.io/ascend/cann:8.5.0-310p-openeuler24.03-py3.11
    docker pull "${source}"
    [[ $(docker image inspect --format '{{.Id}}' "${source}") == "${expected}" ]] || {
        printf 'Base image changed; restore the archived image instead.\n' >&2; exit 1;
    }
    docker tag "${source}" "${base}"
fi
[[ $(docker image inspect --format '{{.Id}}' "${base}") == "${expected}" ]] || {
    printf 'Unexpected base image ID.\n' >&2; exit 1;
}
mkdir -p "${bundle}/wheels"
cp "${root}/docker/requirements-cann-runtime.lock" "${bundle}/"
cp "${root}/docker/Dockerfile.cann-runtime" "${bundle}/Dockerfile"
cp "${root}/docker/runtime-entrypoint.sh" "${bundle}/"
printf '*\n!Dockerfile\n!runtime-entrypoint.sh\n!requirements-cann-runtime.lock\n!wheels/\n!wheels/**\n' > "${bundle}/.dockerignore"
if [[ ${1:-} == --download ]]; then
    docker run --rm --network host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" \
        -e http_proxy -e https_proxy -e no_proxy -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
        -v "${bundle}:/environment" --entrypoint python3 "${base}" \
        -m pip download --no-cache-dir --progress-bar off --require-hashes --only-binary=:all: --no-deps \
        --dest /environment/wheels -r /environment/requirements-cann-runtime.lock
fi
docker build --network none -t "${version}" -t umm-cann:8.5.0-mvp "${bundle}"
image_id=$(docker image inspect --format '{{.Id}}' "${version}")
if docker container inspect "${guard}" >/dev/null 2>&1; then
    if [[ $(docker container inspect --format '{{.Image}}' "${guard}") != "${image_id}" ]]; then
        [[ $(docker container inspect --format '{{.State.Status}}/{{index .Config.Labels "umm.environment.keep"}}' "${guard}") == created/true ]] || {
            printf 'Protection container is not an owned, unstarted guard; retained unchanged.\n' >&2; exit 1;
        }
        docker rm "${guard}"
    fi
fi
if ! docker container inspect "${guard}" >/dev/null 2>&1; then
    docker create --name "${guard}" --label umm.environment.keep=true \
        --entrypoint /bin/true "${version}"
fi
printf 'Ready: %s\nImage ID: %s\nProtection container: %s (do not remove)\n' "${version}" "${image_id}" "${guard}"
