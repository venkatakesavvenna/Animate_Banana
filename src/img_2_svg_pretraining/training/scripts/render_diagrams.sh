#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
diagram_dir="$repo_root/docs/diagrams"

for diagram in "$diagram_dir"/*.mmd; do
    encoded="$(base64 -w0 "$diagram" | tr '+/' '-_' | tr -d '=')"
    output="${diagram%.mmd}.svg"

    if curl --retry 3 --retry-all-errors --retry-delay 1 -fsSL "https://mermaid.ink/svg/$encoded" -o "$output"; then
        continue
    fi

    curl --retry 3 --retry-all-errors --retry-delay 1 -fsSL \
        -H "Content-Type: text/plain" \
        --data-binary "@$diagram" \
        "https://kroki.io/mermaid/svg" \
        -o "$output"
done
