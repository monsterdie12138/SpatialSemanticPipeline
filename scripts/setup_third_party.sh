#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TRANSNET_REPOSITORY="https://github.com/soCzech/TransNetV2.git"
TRANSNET_COMMIT="85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed"
TRANSNET_DIR="$PIPELINE_ROOT/tools/TransNetV2"

VGGT_REPOSITORY="https://github.com/facebookresearch/vggt-omega.git"
VGGT_COMMIT="282ec70363edeff59424bf43731658092fba3d37"
VGGT_DIR="$PIPELINE_ROOT/tools/VGGT-Omega"

clone_at_commit() {
  local repository="$1"
  local commit="$2"
  local target="$3"
  local name="$4"

  if [[ -d "$target/.git" ]]; then
    echo "$name source already exists: $target"
    return
  fi
  if [[ -e "$target" ]]; then
    echo "$name target exists but is not a Git checkout: $target" >&2
    echo "Move it aside or remove it before running this setup script." >&2
    exit 1
  fi

  echo "Cloning $name..."
  GIT_LFS_SKIP_SMUDGE=1 git clone "$repository" "$target"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$target" checkout --detach "$commit"
}

mkdir -p "$PIPELINE_ROOT/tools"
clone_at_commit "$TRANSNET_REPOSITORY" "$TRANSNET_COMMIT" "$TRANSNET_DIR" "TransNetV2"
clone_at_commit "$VGGT_REPOSITORY" "$VGGT_COMMIT" "$VGGT_DIR" "VGGT-Omega"

echo "Third-party source setup completed."
echo "Run 'bash scripts/download_models.sh' to download model weights."
