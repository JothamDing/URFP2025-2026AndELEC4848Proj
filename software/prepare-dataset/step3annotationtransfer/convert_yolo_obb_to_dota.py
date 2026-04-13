#!/usr/bin/env python3
"""Convert YOLO-OBB dataset format to DOTA dataset format.

Expected input structure:
  raw/
    images/
    labels/
    data_obb.yaml

Output structure:
  output/
    train/
      images/
      labelTxt/
    val/
      images/
      labelTxt/
    test/
      images/
      labelTxt/
"""

from __future__ import annotations

import argparse
import random
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import yaml


@dataclass
class ConvertStats:
    total_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    total_groups: int = 0
    train_groups: int = 0
    val_groups: int = 0
    test_groups: int = 0
    written_label_files: int = 0
    copied_images: int = 0
    skipped_bad_lines: int = 0
    skipped_bad_class_id: int = 0
    missing_image_for_label: int = 0


@dataclass
class GroupRecord:
    key: str
    samples: List[Tuple[Path, Path]]
    label_counts: Dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert YOLO-OBB labels to DOTA labelTxt format with balanced "
            "train/val/test split."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("raw"),
        help="Input dataset root containing images/, labels/, and data_obb.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output dataset root for DOTA-format files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config with class names. Defaults to <input-dir>/data_obb.yaml.",
    )
    parser.add_argument(
        "--split-ratios",
        type=str,
        default="0.7,0.15,0.15",
        help=(
            "Comma-separated ratios for train,val,test. "
            "Example: '0.7,0.15,0.15'. Ratios must sum to 1."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling before train/val/test split.",
    )
    parser.add_argument(
        "--difficult",
        type=int,
        default=0,
        choices=[0, 1],
        help="DOTA difficult value appended to each annotation line.",
    )
    parser.add_argument(
        "--image-exts",
        type=str,
        default=".jpg,.jpeg,.png,.bmp,.tif,.tiff",
        help="Comma-separated list of candidate image extensions.",
    )
    return parser.parse_args()


def load_class_names(config_path: Path) -> Dict[int, str]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names")
    if names is None:
        raise ValueError(f"Missing 'names' in config: {config_path}")

    class_map: Dict[int, str] = {}
    if isinstance(names, dict):
        for k, v in names.items():
            class_map[int(k)] = str(v)
    elif isinstance(names, list):
        class_map = {i: str(name) for i, name in enumerate(names)}
    else:
        raise ValueError("'names' must be either a list or a dict in YAML config")

    if not class_map:
        raise ValueError(f"No class names loaded from: {config_path}")
    return class_map


def parse_split_ratios(raw_ratios: str) -> Tuple[float, float, float]:
    parts = [x.strip() for x in raw_ratios.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(
            "--split-ratios must contain exactly 3 values: train,val,test"
        )

    try:
        train_ratio, val_ratio, test_ratio = (float(x) for x in parts)
    except ValueError as exc:
        raise ValueError("--split-ratios values must be numeric") from exc

    for ratio_name, ratio_value in (
        ("train", train_ratio),
        ("val", val_ratio),
        ("test", test_ratio),
    ):
        if not 0.0 <= ratio_value <= 1.0:
            raise ValueError(
                f"{ratio_name} ratio out of range [0, 1]: {ratio_value}"
            )

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            f"--split-ratios must sum to 1.0, got {ratio_sum:.8f}"
        )

    return train_ratio, val_ratio, test_ratio


def collect_samples(
    input_dir: Path, image_exts: Sequence[str], stats: ConvertStats
) -> List[Tuple[Path, Path]]:
    labels_dir = input_dir / "labels"
    images_dir = input_dir / "images"

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    ext_set = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in image_exts}
    samples: List[Tuple[Path, Path]] = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        image_path = None
        for ext in ext_set:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            stats.missing_image_for_label += 1
            continue
        samples.append((image_path, label_path))

    return samples


def get_sample_group_key(label_stem: str) -> str:
    """Use trailing numeric suffix as augmentation index, e.g. 0021-0 -> 0021."""
    prefix, sep, suffix = label_stem.rpartition("-")
    if sep and suffix.isdigit() and prefix:
        return prefix
    return label_stem


def count_groups(samples: Sequence[Tuple[Path, Path]]) -> int:
    return len({get_sample_group_key(label_path.stem) for _, label_path in samples})


def _count_valid_labels_in_file(label_path: Path, valid_class_ids: Set[int]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 9:
            continue
        try:
            class_id = int(parts[0])
            # Keep the same "valid line" requirement as convert_one_label_file.
            _ = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if class_id not in valid_class_ids:
            continue
        counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def _build_group_records(
    samples: Sequence[Tuple[Path, Path]], class_map: Dict[int, str]
) -> List[GroupRecord]:
    grouped_samples: Dict[str, List[Tuple[Path, Path]]] = {}
    for image_path, label_path in samples:
        group_key = get_sample_group_key(label_path.stem)
        grouped_samples.setdefault(group_key, []).append((image_path, label_path))

    valid_class_ids = set(class_map.keys())
    group_records: List[GroupRecord] = []
    for group_key, group_samples in grouped_samples.items():
        label_counts: Dict[int, int] = {}
        for _, label_path in group_samples:
            file_counts = _count_valid_labels_in_file(label_path, valid_class_ids)
            for class_id, count in file_counts.items():
                label_counts[class_id] = label_counts.get(class_id, 0) + count
        group_records.append(
            GroupRecord(
                key=group_key,
                samples=group_samples,
                label_counts=label_counts,
            )
        )
    return group_records


def _calc_split_score(
    train_label_counts: Dict[int, int],
    total_label_counts: Dict[int, int],
    train_sample_count: int,
    total_sample_count: int,
    train_group_count: int,
    total_group_count: int,
    train_ratio: float,
) -> float:
    label_error_sum = 0.0
    active_class_count = 0
    for class_id, total_count in total_label_counts.items():
        if total_count <= 0:
            continue
        active_class_count += 1
        target_train_count = total_count * train_ratio
        observed_train_count = train_label_counts.get(class_id, 0)
        label_error_sum += abs(observed_train_count - target_train_count) / total_count
    label_error = (
        label_error_sum / active_class_count if active_class_count > 0 else 0.0
    )

    sample_error = abs(train_sample_count - total_sample_count * train_ratio) / max(
        1, total_sample_count
    )
    group_error = abs(train_group_count - total_group_count * train_ratio) / max(
        1, total_group_count
    )
    return 0.75 * label_error + 0.20 * sample_error + 0.05 * group_error


def _greedy_group_trial(
    group_records: Sequence[GroupRecord],
    total_label_counts: Dict[int, int],
    train_ratio: float,
    seed: int,
) -> Tuple[Set[str], float]:
    rng = random.Random(seed)
    total_groups = len(group_records)
    total_samples = sum(len(record.samples) for record in group_records)

    rarity_scores: Dict[str, float] = {}
    for record in group_records:
        rarity_score = 0.0
        for class_id, count in record.label_counts.items():
            total_count = total_label_counts.get(class_id, 0)
            if total_count > 0:
                rarity_score += count / total_count
        rarity_scores[record.key] = rarity_score

    ordered_records = sorted(
        group_records,
        key=lambda record: (
            -rarity_scores[record.key],
            -len(record.samples),
            rng.random(),
        ),
    )

    train_keys: Set[str] = set()
    train_label_counts: Dict[int, int] = {}
    train_sample_count = 0
    train_group_count = 0

    for idx, record in enumerate(ordered_records):
        remaining_after = total_groups - idx - 1
        val_group_count = idx - train_group_count

        can_assign_train = True
        can_assign_val = True
        if total_groups > 1:
            # Keep both splits non-empty when there are at least 2 groups.
            if val_group_count == 0 and remaining_after == 0:
                can_assign_train = False
            if train_group_count == 0 and remaining_after == 0:
                can_assign_val = False

        score_if_train = float("inf")
        if can_assign_train:
            candidate_train_counts = dict(train_label_counts)
            for class_id, count in record.label_counts.items():
                candidate_train_counts[class_id] = (
                    candidate_train_counts.get(class_id, 0) + count
                )
            score_if_train = _calc_split_score(
                train_label_counts=candidate_train_counts,
                total_label_counts=total_label_counts,
                train_sample_count=train_sample_count + len(record.samples),
                total_sample_count=total_samples,
                train_group_count=train_group_count + 1,
                total_group_count=total_groups,
                train_ratio=train_ratio,
            )

        score_if_val = float("inf")
        if can_assign_val:
            score_if_val = _calc_split_score(
                train_label_counts=train_label_counts,
                total_label_counts=total_label_counts,
                train_sample_count=train_sample_count,
                total_sample_count=total_samples,
                train_group_count=train_group_count,
                total_group_count=total_groups,
                train_ratio=train_ratio,
            )

        if score_if_train < score_if_val:
            assign_to_train = True
        elif score_if_val < score_if_train:
            assign_to_train = False
        else:
            assign_to_train = can_assign_train and (
                not can_assign_val or rng.random() < 0.5
            )

        if assign_to_train:
            train_keys.add(record.key)
            train_group_count += 1
            train_sample_count += len(record.samples)
            for class_id, count in record.label_counts.items():
                train_label_counts[class_id] = train_label_counts.get(class_id, 0) + count

    final_score = _calc_split_score(
        train_label_counts=train_label_counts,
        total_label_counts=total_label_counts,
        train_sample_count=train_sample_count,
        total_sample_count=total_samples,
        train_group_count=train_group_count,
        total_group_count=total_groups,
        train_ratio=train_ratio,
    )
    return train_keys, final_score


def _apply_count_delta(
    base_counts: Dict[int, int], delta_counts: Dict[int, int], sign: int
) -> None:
    for class_id, count in delta_counts.items():
        new_value = base_counts.get(class_id, 0) + sign * count
        if new_value > 0:
            base_counts[class_id] = new_value
        else:
            base_counts.pop(class_id, None)


def _refine_group_split(
    group_records: Sequence[GroupRecord],
    train_keys: Set[str],
    total_label_counts: Dict[int, int],
    train_ratio: float,
    max_rounds: int = 40,
) -> Set[str]:
    group_map = {record.key: record for record in group_records}
    all_keys = list(group_map.keys())
    total_groups = len(group_records)
    total_samples = sum(len(record.samples) for record in group_records)

    train_label_counts: Dict[int, int] = {}
    train_sample_count = 0
    for key in train_keys:
        record = group_map[key]
        train_sample_count += len(record.samples)
        _apply_count_delta(train_label_counts, record.label_counts, sign=1)

    current_score = _calc_split_score(
        train_label_counts=train_label_counts,
        total_label_counts=total_label_counts,
        train_sample_count=train_sample_count,
        total_sample_count=total_samples,
        train_group_count=len(train_keys),
        total_group_count=total_groups,
        train_ratio=train_ratio,
    )

    for _ in range(max_rounds):
        train_list = [k for k in all_keys if k in train_keys]
        val_list = [k for k in all_keys if k not in train_keys]

        best_score = current_score
        best_action: Optional[Tuple[str, str, str, Dict[int, int], int]] = None

        # Try moving one train group to val.
        if len(train_list) > 1:
            for train_key in train_list:
                train_record = group_map[train_key]
                candidate_counts = dict(train_label_counts)
                _apply_count_delta(candidate_counts, train_record.label_counts, sign=-1)
                candidate_samples = train_sample_count - len(train_record.samples)
                candidate_score = _calc_split_score(
                    train_label_counts=candidate_counts,
                    total_label_counts=total_label_counts,
                    train_sample_count=candidate_samples,
                    total_sample_count=total_samples,
                    train_group_count=len(train_keys) - 1,
                    total_group_count=total_groups,
                    train_ratio=train_ratio,
                )
                if candidate_score < best_score - 1e-12:
                    best_score = candidate_score
                    best_action = (
                        "move_out",
                        train_key,
                        "",
                        candidate_counts,
                        candidate_samples,
                    )

        # Try moving one val group to train.
        if len(val_list) > 1:
            for val_key in val_list:
                val_record = group_map[val_key]
                candidate_counts = dict(train_label_counts)
                _apply_count_delta(candidate_counts, val_record.label_counts, sign=1)
                candidate_samples = train_sample_count + len(val_record.samples)
                candidate_score = _calc_split_score(
                    train_label_counts=candidate_counts,
                    total_label_counts=total_label_counts,
                    train_sample_count=candidate_samples,
                    total_sample_count=total_samples,
                    train_group_count=len(train_keys) + 1,
                    total_group_count=total_groups,
                    train_ratio=train_ratio,
                )
                if candidate_score < best_score - 1e-12:
                    best_score = candidate_score
                    best_action = (
                        "move_in",
                        val_key,
                        "",
                        candidate_counts,
                        candidate_samples,
                    )

        # Try swapping one group between train and val.
        for train_key in train_list:
            train_record = group_map[train_key]
            for val_key in val_list:
                val_record = group_map[val_key]
                candidate_counts = dict(train_label_counts)
                _apply_count_delta(candidate_counts, train_record.label_counts, sign=-1)
                _apply_count_delta(candidate_counts, val_record.label_counts, sign=1)
                candidate_samples = (
                    train_sample_count
                    - len(train_record.samples)
                    + len(val_record.samples)
                )
                candidate_score = _calc_split_score(
                    train_label_counts=candidate_counts,
                    total_label_counts=total_label_counts,
                    train_sample_count=candidate_samples,
                    total_sample_count=total_samples,
                    train_group_count=len(train_keys),
                    total_group_count=total_groups,
                    train_ratio=train_ratio,
                )
                if candidate_score < best_score - 1e-12:
                    best_score = candidate_score
                    best_action = (
                        "swap",
                        train_key,
                        val_key,
                        candidate_counts,
                        candidate_samples,
                    )

        if best_action is None:
            break

        action, src_key, dst_key, next_counts, next_samples = best_action
        if action == "move_out":
            train_keys.remove(src_key)
        elif action == "move_in":
            train_keys.add(src_key)
        elif action == "swap":
            train_keys.remove(src_key)
            train_keys.add(dst_key)
        else:
            raise ValueError(f"Unknown refine action: {action}")

        train_label_counts = next_counts
        train_sample_count = next_samples
        current_score = best_score

    return train_keys


def make_split(
    samples: List[Tuple[Path, Path]],
    train_ratio: float,
    seed: int,
    class_map: Dict[int, str],
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio must be in [0, 1]")

    group_records = _build_group_records(samples, class_map)
    if not group_records:
        return [], []

    if len(group_records) == 1:
        only_group_samples = list(group_records[0].samples)
        if train_ratio >= 0.5:
            return only_group_samples, []
        return [], only_group_samples

    total_label_counts: Dict[int, int] = {}
    for record in group_records:
        for class_id, count in record.label_counts.items():
            total_label_counts[class_id] = total_label_counts.get(class_id, 0) + count

    master_rng = random.Random(seed)
    trial_count = min(256, max(32, len(group_records) * 4))

    best_train_keys: Set[str] = set()
    best_score = float("inf")
    for _ in range(trial_count):
        trial_seed = master_rng.randrange(0, 2**31)
        train_keys, score = _greedy_group_trial(
            group_records=group_records,
            total_label_counts=total_label_counts,
            train_ratio=train_ratio,
            seed=trial_seed,
        )
        if score < best_score:
            best_score = score
            best_train_keys = train_keys

    best_train_keys = _refine_group_split(
        group_records=group_records,
        train_keys=set(best_train_keys),
        total_label_counts=total_label_counts,
        train_ratio=train_ratio,
    )

    train_samples: List[Tuple[Path, Path]] = []
    val_samples: List[Tuple[Path, Path]] = []
    for record in group_records:
        target = train_samples if record.key in best_train_keys else val_samples
        target.extend(record.samples)

    return train_samples, val_samples


def make_three_way_split(
    samples: List[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    class_map: Dict[int, str],
) -> Tuple[
    List[Tuple[Path, Path]],
    List[Tuple[Path, Path]],
    List[Tuple[Path, Path]],
]:
    if not samples:
        return [], [], []

    train_samples, holdout_samples = make_split(
        samples=samples,
        train_ratio=train_ratio,
        seed=seed,
        class_map=class_map,
    )

    holdout_ratio = val_ratio + test_ratio
    if holdout_ratio <= 0.0:
        return train_samples, [], []
    if not holdout_samples:
        return train_samples, [], []

    # Split the holdout set into val/test while keeping annotation balance.
    val_within_holdout_ratio = val_ratio / holdout_ratio
    val_samples, test_samples = make_split(
        samples=holdout_samples,
        train_ratio=val_within_holdout_ratio,
        seed=seed + 1,
        class_map=class_map,
    )
    return train_samples, val_samples, test_samples


def _collect_split_label_counts(
    samples: Sequence[Tuple[Path, Path]], class_map: Dict[int, str]
) -> Dict[int, int]:
    valid_class_ids = set(class_map.keys())
    split_counts: Dict[int, int] = {}
    for _, label_path in samples:
        file_counts = _count_valid_labels_in_file(label_path, valid_class_ids)
        for class_id, count in file_counts.items():
            split_counts[class_id] = split_counts.get(class_id, 0) + count
    return split_counts


def convert_one_label_file(
    label_path: Path,
    width: int,
    height: int,
    class_map: Dict[int, str],
    difficult: int,
    stats: ConvertStats,
) -> List[str]:
    lines_out: List[str] = []
    raw_lines = label_path.read_text(encoding="utf-8").splitlines()

    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue

        parts = raw.split()
        if len(parts) != 9:
            stats.skipped_bad_lines += 1
            continue

        try:
            class_id = int(parts[0])
            pts = [float(x) for x in parts[1:]]
        except ValueError:
            stats.skipped_bad_lines += 1
            continue

        class_name = class_map.get(class_id)
        if class_name is None:
            stats.skipped_bad_class_id += 1
            continue

        abs_pts: List[float] = []
        for idx, value in enumerate(pts):
            if idx % 2 == 0:
                abs_pts.append(value * width)
            else:
                abs_pts.append(value * height)

        coord_str = " ".join(f"{v:.6f}" for v in abs_pts)
        lines_out.append(f"{coord_str} {class_name} {difficult}")

    return lines_out


def ensure_output_dirs(output_dir: Path) -> Dict[str, Dict[str, Path]]:
    layout = {
        "train": {
            "images": output_dir / "train" / "images",
            "labelTxt": output_dir / "train" / "labelTxt",
        },
        "val": {
            "images": output_dir / "val" / "images",
            "labelTxt": output_dir / "val" / "labelTxt",
        },
        "test": {
            "images": output_dir / "test" / "images",
            "labelTxt": output_dir / "test" / "labelTxt",
        },
    }
    for split in layout.values():
        for p in split.values():
            p.mkdir(parents=True, exist_ok=True)
    return layout


def process_split(
    split_name: str,
    samples: Sequence[Tuple[Path, Path]],
    dirs: Dict[str, Path],
    class_map: Dict[int, str],
    difficult: int,
    stats: ConvertStats,
) -> None:
    for image_path, label_path in samples:
        width, height = get_image_size(image_path)

        dota_lines = convert_one_label_file(
            label_path=label_path,
            width=width,
            height=height,
            class_map=class_map,
            difficult=difficult,
            stats=stats,
        )

        out_label = dirs["labelTxt"] / f"{label_path.stem}.txt"
        out_label.write_text("\n".join(dota_lines) + ("\n" if dota_lines else ""), encoding="utf-8")
        stats.written_label_files += 1

        out_image = dirs["images"] / image_path.name
        shutil.copy2(image_path, out_image)
        stats.copied_images += 1

    if split_name == "train":
        stats.train_samples = len(samples)
    elif split_name == "val":
        stats.val_samples = len(samples)
    elif split_name == "test":
        stats.test_samples = len(samples)


def get_image_size(image_path: Path) -> Tuple[int, int]:
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return _get_jpeg_size(image_path)
    if suffix == ".png":
        return _get_png_size(image_path)
    raise ValueError(f"Unsupported image format for size parsing: {image_path}")


def _get_png_size(image_path: Path) -> Tuple[int, int]:
    with image_path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG file: {image_path}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def _get_jpeg_size(image_path: Path) -> Tuple[int, int]:
    with image_path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError(f"Invalid JPEG file: {image_path}")

        while True:
            marker_start = f.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue

            marker_code = f.read(1)
            while marker_code == b"\xff":
                marker_code = f.read(1)
            if not marker_code:
                break

            # Start of Scan or End of Image.
            if marker_code in {b"\xda", b"\xd9"}:
                break

            segment_length_bytes = f.read(2)
            if len(segment_length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", segment_length_bytes)[0]
            if segment_length < 2:
                raise ValueError(f"Corrupted JPEG segment in: {image_path}")

            # SOF markers contain image size.
            if marker_code in {
                b"\xc0",
                b"\xc1",
                b"\xc2",
                b"\xc3",
                b"\xc5",
                b"\xc6",
                b"\xc7",
                b"\xc9",
                b"\xca",
                b"\xcb",
                b"\xcd",
                b"\xce",
                b"\xcf",
            }:
                sof_data = f.read(segment_length - 2)
                if len(sof_data) < 5:
                    break
                height, width = struct.unpack(">HH", sof_data[1:5])
                return int(width), int(height)

            f.seek(segment_length - 2, 1)

    raise ValueError(f"Failed to read JPEG size: {image_path}")


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    config_path = args.config if args.config is not None else input_dir / "data_obb.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    image_exts = [x.strip() for x in args.image_exts.split(",") if x.strip()]
    train_ratio, val_ratio, test_ratio = parse_split_ratios(args.split_ratios)
    stats = ConvertStats()

    class_map = load_class_names(config_path)
    samples = collect_samples(input_dir=input_dir, image_exts=image_exts, stats=stats)
    stats.total_samples = len(samples)

    train_samples, val_samples, test_samples = make_three_way_split(
        samples=samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=args.seed,
        class_map=class_map,
    )
    stats.total_groups = count_groups(samples)
    stats.train_groups = count_groups(train_samples)
    stats.val_groups = count_groups(val_samples)
    stats.test_groups = count_groups(test_samples)
    out_dirs = ensure_output_dirs(output_dir)

    process_split(
        split_name="train",
        samples=train_samples,
        dirs=out_dirs["train"],
        class_map=class_map,
        difficult=args.difficult,
        stats=stats,
    )
    process_split(
        split_name="val",
        samples=val_samples,
        dirs=out_dirs["val"],
        class_map=class_map,
        difficult=args.difficult,
        stats=stats,
    )
    process_split(
        split_name="test",
        samples=test_samples,
        dirs=out_dirs["test"],
        class_map=class_map,
        difficult=args.difficult,
        stats=stats,
    )

    print("Conversion complete")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Class count: {len(class_map)}")
    print(f"Total samples used: {stats.total_samples}")
    print(
        "Target split ratios: "
        f"train={train_ratio:.3f}, val={val_ratio:.3f}, test={test_ratio:.3f}"
    )
    print(f"Train samples: {stats.train_samples}")
    print(f"Val samples: {stats.val_samples}")
    print(f"Test samples: {stats.test_samples}")
    print(f"Total groups used: {stats.total_groups}")
    print(f"Train groups: {stats.train_groups}")
    print(f"Val groups: {stats.val_groups}")
    print(f"Test groups: {stats.test_groups}")
    print(f"Images copied: {stats.copied_images}")
    print(f"Label files written: {stats.written_label_files}")
    print(f"Missing-image labels skipped: {stats.missing_image_for_label}")
    print(f"Bad-format lines skipped: {stats.skipped_bad_lines}")
    print(f"Bad-class-id lines skipped: {stats.skipped_bad_class_id}")

    train_label_counts = _collect_split_label_counts(train_samples, class_map)
    val_label_counts = _collect_split_label_counts(val_samples, class_map)
    test_label_counts = _collect_split_label_counts(test_samples, class_map)
    print("Per-class split distribution:")
    for class_id, class_name in sorted(class_map.items()):
        train_count = train_label_counts.get(class_id, 0)
        val_count = val_label_counts.get(class_id, 0)
        test_count = test_label_counts.get(class_id, 0)
        total_count = train_count + val_count + test_count
        if total_count == 0:
            continue
        train_ratio_actual = train_count / total_count
        val_ratio_actual = val_count / total_count
        test_ratio_actual = test_count / total_count
        train_diff = train_ratio_actual - train_ratio
        val_diff = val_ratio_actual - val_ratio
        test_diff = test_ratio_actual - test_ratio
        print(
            f"  [{class_id}] {class_name}: "
            f"train={train_count}, val={val_count}, test={test_count}, "
            f"train_ratio={train_ratio_actual:.3f} (target={train_ratio:.3f}, diff={train_diff:+.3f}), "
            f"val_ratio={val_ratio_actual:.3f} (target={val_ratio:.3f}, diff={val_diff:+.3f}), "
            f"test_ratio={test_ratio_actual:.3f} (target={test_ratio:.3f}, diff={test_diff:+.3f})"
        )


if __name__ == "__main__":
    main()
