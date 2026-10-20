#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
bundle=${root}/artifacts/runtime/cann310p-20261020
archive=${bundle}/umm-cann-runtime-20261020.tar.gz
[[ ! -e ${archive} && ! -e ${archive}.partial ]] || {
    printf 'Archive or partial archive already exists: %s\n' "${archive}" >&2; exit 1;
}
mkdir -p "${bundle}"
docker save umm-cann:8.5.0-mvp umm-cann:8.5.0-310p-runtime-20261020 \
    umm-cann-base:8.5.0-310p-20261020 | gzip -1 > "${archive}.partial"
gzip -t "${archive}.partial"
mv "${archive}.partial" "${archive}"
(cd "${bundle}" && sha256sum "$(basename -- "${archive}")" > "$(basename -- "${archive}").sha256")
printf 'Saved: %s\n' "${archive}"
