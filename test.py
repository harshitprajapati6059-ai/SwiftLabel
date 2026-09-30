"""
test.py - Run the full pipeline on a single image and print a readable report.

Usage:
    python test.py images/sample.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import config
from utils import run_pipeline


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python test.py <path_to_image>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    try:
        result = run_pipeline(image_path)
    except Exception as exc:
        print(f"✗ Failed: {exc}")
        sys.exit(1)

    print("✓ Surface segmentation completed")
    print("✓ Per-class masks cleaned (ego vehicle excluded)")
    print(f"✓ {result.total_regions} polygon(s) extracted")
    print("✓ Overlay generated")
    print("✓ JSON exported")
    print()
    print(f"Processing time       : {result.processing_time_sec}s")
    print(f"Drivable area         : {result.drivable_area_pct}%")
    print()
    print(f"{'CLASS':<12} {'DRIVABLE':<9} {'POLYGONS':>8} {'COVERAGE':>9} {'CONF':>6}")
    for surface in config.SURFACE_CLASSES:
        count = result.region_counts.get(surface.key, 0)
        coverage = result.coverage_pct.get(surface.key, 0.0)
        confidence = result.confidences.get(surface.key)
        confidence_text = f"{confidence:.2f}" if confidence is not None else "-"
        print(
            f"{surface.key:<12} {('yes' if surface.drivable else 'no'):<9} "
            f"{count:>8} {coverage:>8.2f}% {confidence_text:>6}"
        )
    print()
    print(f"Colour mask saved to  : {result.mask_path}")
    print(f"Label map saved to    : {result.label_map_path}")
    print(f"Overlay saved to      : {result.overlay_path}")
    print(f"Polygon JSON saved to : {result.polygon_path}")
    print(f"Metadata saved to     : {result.metadata_path}")

    if result.warnings:
        print()
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
