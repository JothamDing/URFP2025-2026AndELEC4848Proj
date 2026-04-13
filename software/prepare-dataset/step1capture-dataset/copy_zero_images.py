import argparse
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ZERO_SUFFIX_PATTERN = re.compile(r".+-0$", re.IGNORECASE)


def resolve_source_folder(src_arg: str | None) -> Path:
    if src_arg:
        return Path(src_arg)

    preferred_sources = [Path("capture image"), Path("captured_images")]
    for folder in preferred_sources:
        if folder.exists() and folder.is_dir():
            return folder

    return preferred_sources[0]


def copy_zero_images(source_dir: Path, target_dir: Path) -> int:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in sorted(source_dir.iterdir()):
        if not item.is_file():
            continue
        if item.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not ZERO_SUFFIX_PATTERN.match(item.stem):
            continue

        shutil.copy2(item, target_dir / item.name)
        copied += 1

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy images whose name ends with '-0' into raw-image folder without renaming."
    )
    parser.add_argument("--src", help="Source folder path")
    parser.add_argument("--dst", default="raw-image", help="Target folder path (default: raw-image)")
    args = parser.parse_args()

    source_dir = resolve_source_folder(args.src)
    target_dir = Path(args.dst)

    copied = copy_zero_images(source_dir, target_dir)
    print(f"Done. Copied {copied} file(s) from '{source_dir}' to '{target_dir}'.")


if __name__ == "__main__":
    main()
