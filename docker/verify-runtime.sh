#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
expected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' "${root}/docker/runtime-cann310p.lock.json")
[[ $(docker image inspect --format '{{.Id}}' umm-cann:8.5.0-mvp) == "${expected}" ]] || {
    printf 'Runtime Image ID differs from the recorded environment.\n' >&2; exit 1;
}
docker run --rm -i --network none --security-opt seccomp=unconfined \
    umm-cann:8.5.0-mvp bash -lc 'test "$(umask)" = 0022; exec python3 -' <<'PY'
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import tempfile

versions = {}
for line in Path('/opt/umm-environment/requirements.lock').read_text().splitlines():
    name, expected = line.split()[0].split('==')
    actual = importlib.metadata.version(name)
    assert actual == expected, (name, actual, expected)
    versions[name] = actual
for module in ('absl', 'attrs', 'cloudpickle', 'decorator', 'jinja2', 'markupsafe',
               'ml_dtypes', 'mpmath', 'numpy', 'packaging', 'PIL', 'google.protobuf',
               'psutil', 'yaml', 'scipy', 'sympy', 'tornado', 'tqdm', 'typing_extensions',
               'getopt', 'inspect', 'multiprocessing'):
    importlib.import_module(module)
check = subprocess.run(['python3', '-m', 'pip', 'check'], text=True, capture_output=True)
known = {'op-compile-tool 0.1.0 requires '+name+', which is not installed.'
         for name in ('getopt', 'inspect', 'multiprocessing')}
assert check.returncode == 0 or (check.returncode == 1 and set(check.stdout.splitlines()) == known), check.stdout+check.stderr
with tempfile.TemporaryDirectory() as directory:
    source = Path(directory)
    (source/'main.cpp').write_text('#include <iostream>\nint main(){std::cout << "umm-runtime";}\n')
    (source/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.20)\nproject(runtime_smoke CXX)\nadd_executable(runtime_smoke main.cpp)\n')
    subprocess.run(['cmake', '-S', directory, '-B', directory+'/build'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['cmake', '--build', directory+'/build', '-j', '2'], check=True, stdout=subprocess.DEVNULL)
    assert subprocess.check_output([directory+'/build/runtime_smoke'], text=True) == 'umm-runtime'
print(json.dumps({'status': 'passed', 'python': platform.python_version(),
                  'architecture': platform.machine(), 'packages': versions,
                  'cmake_cpp_build_and_run': 'passed',
                  'known_vendor_metadata_notices': sorted(known) if check.returncode else []}, indent=2))
PY
