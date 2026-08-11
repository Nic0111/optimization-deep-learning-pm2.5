#!/usr/bin/env bash
set -euo pipefail

# Prepare the Linux-side project without installing a system-wide Python stack.
if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "This script must run inside the Ubuntu WSL distribution." >&2
  exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$project_root" in
  /mnt/*)
    echo "The active project must live under the WSL home directory, not /mnt/c or /mnt/g." >&2
    exit 1
    ;;
esac

required_packages=(python3-venv python3-pip curl git tmux)
missing_packages=()
for package in "${required_packages[@]}"; do
  if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
    missing_packages+=("$package")
  fi
done
if (( ${#missing_packages[@]} > 0 )); then
  echo "Installing missing Ubuntu packages: ${missing_packages[*]}"
  sudo apt-get update
  sudo apt-get install -y "${missing_packages[@]}"
fi

uv_home="$HOME/.local/share/odl-uv"
uv_binary="$HOME/.local/bin/uv"
if [[ ! -x "$uv_binary" ]]; then
  echo "Installing the pinned uv environment manager..."
  python3 -m venv "$uv_home"
  "$uv_home/bin/python" -m pip install --upgrade pip
  "$uv_home/bin/python" -m pip install "uv==0.11.3"
  mkdir -p "$HOME/.local/bin"
  ln -sfn "$uv_home/bin/uv" "$uv_binary"
fi

cd "$project_root"
"$uv_binary" sync --all-groups --frozen
chmod +x scripts/python_wsl.sh scripts/odl.sh
"$uv_binary" run python -m ipykernel install --user \
  --name odl-assignment \
  --display-name "Python (odl-assignment)"

# VS Code/Jupyter must start the kernel through the same CUDA-aware launcher.
kernel_json="$HOME/.local/share/jupyter/kernels/odl-assignment/kernel.json"
"$uv_binary" run python - "$kernel_json" "$project_root/scripts/python_wsl.sh" <<'PY'
import json
import sys
from pathlib import Path

kernel_path = Path(sys.argv[1])
launcher = sys.argv[2]
payload = json.loads(kernel_path.read_text(encoding="utf-8"))
payload["argv"][0] = launcher
kernel_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

gpu_test='import tensorflow as tf; gpus=tf.config.list_physical_devices("GPU"); print(gpus); raise SystemExit(0 if gpus else 1)'
if ! "$project_root/scripts/python_wsl.sh" -c "$gpu_test"; then
  echo "TensorFlow cannot use the WSL GPU even though the locked CUDA environment is installed." >&2
  exit 1
fi

echo "Environment created at $project_root/.venv"
"$project_root/scripts/odl.sh" preflight --mode smoke --device gpu
