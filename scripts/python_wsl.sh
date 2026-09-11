#!/usr/bin/env bash
set -euo pipefail

# Launch the project interpreter with CUDA libraries installed by
# tensorflow[and-cuda] visible to Linux's dynamic loader.
if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "This launcher must run inside the Ubuntu WSL distribution." >&2
  exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_binary="$project_root/.venv/bin/python"
nvidia_root="$project_root/.venv/lib/python3.12/site-packages/nvidia"

if [[ ! -x "$python_binary" ]]; then
  echo "Project environment not found. Run scripts/bootstrap_wsl.sh first." >&2
  exit 1
fi

cuda_library_path=""
if [[ -d "$nvidia_root" ]]; then
  while IFS= read -r library_dir; do
    if [[ -z "$cuda_library_path" ]]; then
      cuda_library_path="$library_dir"
    else
      cuda_library_path="$cuda_library_path:$library_dir"
    fi
  done < <(find "$nvidia_root" -type d -name lib -print | sort)
fi

if [[ -n "$cuda_library_path" ]]; then
  export LD_LIBRARY_PATH="$cuda_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

ptxas_dir="$nvidia_root/cuda_nvcc/bin"
if [[ -d "$ptxas_dir" ]]; then
  export PATH="$ptxas_dir:$PATH"
fi

exec "$python_binary" "$@"
