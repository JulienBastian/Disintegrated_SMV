#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
repo_root="$(cd -- "$script_dir/.." >/dev/null 2>&1 && pwd)"
cd "$repo_root"

python3 -m compileall -q core data models third_party scripts cbound.py training.py training_variability.py optimization.py

bash -n training.sh
bash -n variability.sh

echo "Static checks passed."
