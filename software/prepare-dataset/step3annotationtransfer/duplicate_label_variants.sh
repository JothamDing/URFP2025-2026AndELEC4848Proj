#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./duplicate_label_variants.sh [PATH] [--force]

Description:
  For each *-0.txt, create:
    *-1.txt
    *-2.txt
  with the exact same content.

Defaults:
  PATH = ./raw/labels (relative to script location)

PATH can be either:
  1) A single label directory (e.g. ./raw/labels or ./output/train/labelTxt)
  2) A dataset root containing split label dirs:
       ./output/train/labelTxt
       ./output/val/labelTxt
       ./output/test/labelTxt

Options:
  --force   Overwrite existing *-1.txt / *-2.txt
  -h, --help  Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_PATH=""
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --force)
      FORCE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$TARGET_PATH" ]]; then
        TARGET_PATH="$arg"
      else
        echo "Unexpected argument: $arg" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$TARGET_PATH" ]]; then
  TARGET_PATH="$SCRIPT_DIR/raw/labels"
fi

if [[ ! -d "$TARGET_PATH" ]]; then
  echo "Path not found: $TARGET_PATH" >&2
  exit 1
fi

duplicate_one_dir() {
  local label_dir="$1"
  local created_count=0
  local skipped_count=0
  local src base_name prefix dst idx

  shopt -s nullglob
  local source_files=("$label_dir"/*-0.txt)
  shopt -u nullglob

  if [[ ${#source_files[@]} -eq 0 ]]; then
    echo "No *-0.txt files found in: $label_dir"
    echo "Done for $label_dir. Created: 0, Skipped: 0, Sources: 0"
    DUP_CREATED=0
    DUP_SKIPPED=0
    DUP_SOURCES=0
    return 0
  fi

  for src in "${source_files[@]}"; do
    base_name="$(basename "$src")"
    prefix="${base_name%-0.txt}"

    for idx in 1 2; do
      dst="$label_dir/${prefix}-${idx}.txt"

      if [[ -e "$dst" && "$FORCE" -ne 1 ]]; then
        echo "Skip existing: $dst"
        skipped_count=$((skipped_count + 1))
        continue
      fi

      cp -f -- "$src" "$dst"
      echo "Created: $dst"
      created_count=$((created_count + 1))
    done
  done

  echo "Done for $label_dir. Created: $created_count, Skipped: $skipped_count, Sources: ${#source_files[@]}"
  DUP_CREATED="$created_count"
  DUP_SKIPPED="$skipped_count"
  DUP_SOURCES="${#source_files[@]}"
}

label_dirs=()
for split in train val test; do
  split_dir="$TARGET_PATH/$split/labelTxt"
  if [[ -d "$split_dir" ]]; then
    label_dirs+=("$split_dir")
  fi
done

if [[ ${#label_dirs[@]} -eq 0 ]]; then
  label_dirs=("$TARGET_PATH")
fi

total_created=0
total_skipped=0
total_sources=0

for label_dir in "${label_dirs[@]}"; do
  duplicate_one_dir "$label_dir"
  total_created=$((total_created + DUP_CREATED))
  total_skipped=$((total_skipped + DUP_SKIPPED))
  total_sources=$((total_sources + DUP_SOURCES))
done

echo "All done. Dirs: ${#label_dirs[@]}, Created: $total_created, Skipped: $total_skipped, Sources: $total_sources"
