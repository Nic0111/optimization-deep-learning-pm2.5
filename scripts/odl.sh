#!/usr/bin/env bash
set -euo pipefail

# Stable WSL entry point for preflight, smoke tests, training, and status.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$project_root/scripts/python_wsl.sh" "$project_root/scripts/odl.py" "$@"
