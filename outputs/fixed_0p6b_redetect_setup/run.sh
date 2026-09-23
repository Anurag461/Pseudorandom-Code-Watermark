#!/usr/bin/env bash
set -euo pipefail

# The default performs CPU preflight; full explicitly starts replay and scoring.
stage="${1:-preflight}"
group="${2:-main}"
case "$stage" in preflight|smoke|full) ;; *) echo "Use preflight, smoke, or full" >&2; exit 2 ;; esac

cd -- "$(dirname -- "$0")/../.."
export MODAL_PROFILE=new-prc-watermark
export NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=1
python_bin="${PRC_REDETECT_PYTHON:-python}"

exec "$python_bin" - "$stage" "$group" <<'PYTHON'
import hashlib, json, subprocess, sys
from pathlib import Path
root = Path('outputs/fixed_0p6b_redetect_setup')
index = json.loads((root / 'index.json').read_text())
stage, group = sys.argv[1:]
selected = [(name, item) for name, item in index['manifests'].items()
            if group == 'all' or item['group'] == group or name == group]
if not selected:
    raise SystemExit('Use main, replicates, all, or an exact manifest name from index.json')
for name, item in selected:
    path = Path(item['path'])
    if not index['cpu_preflight_passed'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        raise SystemExit('Frozen manifest is missing, changed, or not fully CPU-verified')
for name, item in selected:
    subprocess.run([sys.executable, '-m', 'modal', 'run', '--detach', 'modal_run.py::redetect',
                    '--manifest', item['path'], '--stage', stage, '--gpu', item['gpu'],
                    '--max-containers', '10', '--csv-out', index['csv_out']], check=True)
PYTHON
