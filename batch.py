"""
batch.py - Process every supported image in a folder.

Usage:
    python batch.py ./images
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from utils import run_pipeline, is_supported_image, get_logger

logger = get_logger()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python batch.py <path_to_folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"✗ Not a folder: {folder}")
        sys.exit(1)

    all_files = sorted(folder.iterdir())
    image_files = [f for f in all_files if f.is_file() and is_supported_image(f)]
    skipped = [f for f in all_files if f.is_file() and not is_supported_image(f)]

    print(f"Found {len(image_files)} supported image(s), skipping {len(skipped)} file(s).")

    start = time.time()
    succeeded = 0
    failed = 0

    for i, image_path in enumerate(image_files, start=1):
        print(f"[{i}/{len(image_files)}] Processing {image_path.name} ...", end=" ")
        try:
            result = run_pipeline(image_path)
            counts = " ".join(f"{k}={v}" for k, v in result.region_counts.items())
            print(f"OK ({result.processing_time_sec}s, {counts})")
            succeeded += 1
        except Exception as exc:
            print(f"FAILED: {exc}")
            logger.error(f"{image_path.name}: {exc}")
            failed += 1

    total_time = round(time.time() - start, 2)
    print()
    print(f"Done. Succeeded: {succeeded}, Failed: {failed}, Skipped: {len(skipped)}")
    print(f"Total processing time: {total_time}s")


if __name__ == "__main__":
    main()
    