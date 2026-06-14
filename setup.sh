#!/usr/bin/env bash
# microserve setup + test gate for a freshly rented GPU box.
#
# Usage (from the project root, after `git clone`):
#   bash setup.sh
#
# On success: tests pass and you can terminate the instance.
# On failure: capture the error, terminate the instance, debug locally,
# push the fix, re-rent. Don't pay rental time staring at stack traces.

set -euo pipefail

# ---- 1. Sanity: project root ----
if [[ ! -f pyproject.toml ]]; then
    echo "ERROR: must run from project root (pyproject.toml not found in $(pwd))"
    exit 1
fi

# ---- 2. GPU + driver visible ----
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. No NVIDIA driver installed, or wrong machine."
    exit 1
fi
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
echo

# ---- 3. Create venv (idempotent) ----
if [[ ! -d .venv ]]; then
    echo "=== Creating venv ==="
    if ! python3 -m venv .venv 2>/dev/null; then
        echo "python3 -m venv failed; installing python3-venv..."
        sudo apt-get update -qq
        sudo apt-get install -y python3-venv
        python3 -m venv .venv
    fi
fi

# ---- 4. Activate + install ----
# shellcheck source=/dev/null
source .venv/bin/activate
echo "=== Installing microserve and dev deps ==="
pip install --quiet --upgrade pip
pip install -e ".[dev]"
echo

# ---- 5. Verify torch sees CUDA before paying for a test run ----
echo "=== Verifying torch ==="
python - <<'PY'
import torch
assert torch.cuda.is_available(), (
    f"torch={torch.__version__} does not see CUDA. "
    "Likely a CPU-only wheel was installed. "
    "Try: pip install torch --index-url https://download.pytorch.org/whl/cu121"
)
print(f"torch={torch.__version__}  cuda={torch.version.cuda}  gpu={torch.cuda.get_device_name(0)}")
PY
echo

# ---- 6. Run the test gate ----
echo "=== Running tests ==="
pytest tests/ -v

echo
echo "=== Done. Tests passed. Safe to terminate the instance. ==="
