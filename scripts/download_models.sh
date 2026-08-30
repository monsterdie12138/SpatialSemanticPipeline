#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TRANSNET_DIR="$PIPELINE_ROOT/tools/TransNetV2/inference-pytorch"
TRANSNET_TARGET="$TRANSNET_DIR/transnetv2-pytorch-weights.pth"
TRANSNET_REPOSITORY="MiaoshouAI/transnetv2-pytorch-weights"
TRANSNET_SHA256="46520d66d4bf60414a4d82e0e94a92442ff950e34517a3718b2e54815e642b53"
VGGT_DIR="$PIPELINE_ROOT/tools/VGGT-Omega/checkpoints/VGGT-Omega-1B-512"
VGGT_TARGET="$VGGT_DIR/model.pt"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf' CLI. Install huggingface_hub and authenticate before downloading." >&2
  exit 1
fi

if [[ ! -d "$TRANSNET_DIR" || ! -d "$PIPELINE_ROOT/tools/VGGT-Omega" ]]; then
  echo "Third-party source is missing. Run 'bash scripts/setup_third_party.sh' first." >&2
  exit 1
fi

if [[ -f "$TRANSNET_TARGET" ]]; then
  echo "TransNetV2 checkpoint already exists: $TRANSNET_TARGET"
else
  hf download "$TRANSNET_REPOSITORY" transnetv2-pytorch-weights.pth --local-dir "$TRANSNET_DIR"
  printf '%s  %s\n' "$TRANSNET_SHA256" "$TRANSNET_TARGET" | sha256sum --check --status
  echo "Downloaded and verified TransNetV2 checkpoint: $TRANSNET_TARGET"
fi

if [[ -f "$VGGT_TARGET" ]]; then
  echo "VGGT-Omega checkpoint already exists: $VGGT_TARGET"
else
  mkdir -p "$VGGT_DIR"
  hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir "$VGGT_DIR"
  mv "$VGGT_DIR/vggt_omega_1b_512.pt" "$VGGT_TARGET"
  echo "Downloaded VGGT-Omega checkpoint to $VGGT_TARGET"
fi
