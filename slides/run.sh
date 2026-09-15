#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v presenterm >/dev/null 2>&1; then
    echo "Error: cal instal·lar presenterm i tenir-lo al PATH." >&2
    exit 127
fi

exec presenterm "$script_dir/slides.md" "$@"
