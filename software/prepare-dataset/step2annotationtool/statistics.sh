#!/usr/bin/env bash
set -euo pipefail

yaml_path="${1:-datasets/data_obb.yaml}"
labels_dir="${2:-datasets/labels}"

if [[ ! -f "$yaml_path" ]]; then
  echo "Error: YAML file not found: $yaml_path" >&2
  exit 1
fi

if [[ ! -d "$labels_dir" ]]; then
  echo "Error: labels directory not found: $labels_dir" >&2
  exit 1
fi

nc="$(awk -F': *' '/^nc:[[:space:]]*/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$yaml_path")"

if [[ -z "$nc" || ! "$nc" =~ ^[0-9]+$ ]]; then
  echo "Error: failed to parse numeric 'nc' from $yaml_path" >&2
  exit 1
fi

declare -a names
declare -a counts

for ((i = 0; i < nc; i++)); do
  names[i]="unnamed_${i}"
  counts[i]=0
done

while IFS=$'\t' read -r idx name; do
  [[ -z "${idx:-}" ]] && continue
  if ((idx >= 0 && idx < nc)); then
    names[idx]="$name"
  fi
done < <(
  awk '
    /^names:[[:space:]]*$/ {in_names=1; next}
    in_names==1 {
      if ($0 ~ /^[^[:space:]]/) exit
      if (match($0, /^[[:space:]]*([0-9]+):[[:space:]]*(.*)[[:space:]]*$/, m)) {
        name = m[2]
        gsub(/^["\047]/, "", name)
        gsub(/["\047]$/, "", name)
        print m[1] "\t" name
      }
    }
  ' "$yaml_path"
)

unknown_total=0

shopt -s nullglob
label_files=("$labels_dir"/*.txt)

if ((${#label_files[@]} > 0)); then
  while read -r label_id label_count; do
    [[ -z "${label_id:-}" ]] && continue
    if ((label_id >= 0 && label_id < nc)); then
      counts[label_id]="$label_count"
    else
      unknown_total=$((unknown_total + label_count))
    fi
  done < <(
    awk 'NF > 0 && $1 ~ /^[0-9]+$/ {c[$1]++} END {for (k in c) print k, c[k]}' "${label_files[@]}" | sort -n
  )
fi

total=0

echo "YAML: $yaml_path"
echo "Labels dir: $labels_dir"
echo "nc (class count): $nc"
echo
printf "%-6s %-20s %s\n" "ID" "Name" "Annotations"
printf "%-6s %-20s %s\n" "----" "--------------------" "-----------"

for ((i = 0; i < nc; i++)); do
  c="${counts[i]:-0}"
  printf "%-6d %-20s %d\n" "$i" "${names[i]}" "$c"
  total=$((total + c))
done

echo
echo "Total annotations: $total"
if ((unknown_total > 0)); then
  echo "Out-of-range label id annotations: $unknown_total"
fi
