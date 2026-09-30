"""
export_cvat.py - Emit a CVAT-importable annotation file for every image.

Produces "CVAT for images 1.1" XML (output/cvat_annotations.xml). That format
is the right fit here for three reasons: it stores polygons natively, it keys
each annotation block to a filename via <image name="...">, so CVAT auto-applies
each block to the matching frame in your task, and it needs no per-image id
bookkeeping to line up.

Runs fully offline: no model pass is needed. Each image takes one of two paths.

  * REVIEWED IMAGES are exported verbatim from output/edited/<name>_polygon.json,
    the human-owned copy written by the review UI. The polygon goes out exactly
    as it was drawn. Re-deriving it from the cached mask instead would run the
    contour approximation again and move every vertex the reviewer placed by
    hand — which is the whole reason that file exists.

  * EVERYTHING ELSE falls back to the cached label maps in output/masks/
    (<name>_labels.png, written by a previous run). Keeping this path means the
    polygon-shaping settings (POLYGON_EPSILON_RATIO, MIN_CONTOUR_AREA_PX) still
    apply at export time, so they can be re-tuned without re-running the model.
    Regenerate the maps with `python batch.py ./images` if the images change.

Scope controls which images are included:

    python export_cvat.py                     # every annotated image (default)
    python export_cvat.py --scope approved    # only images signed off in the UI

The UI's export dialog defaults to `approved`; the CLI defaults to `annotated`
so that a plain batch.py -> export_cvat.py run still exports what it always did.

IMPORTANT — labels must exist in the CVAT task first. CVAT rejects any polygon
whose label is not already defined. The labels used are exactly the class keys
in config.SURFACE_CLASSES (road, bike_path, walkway, green_area); add any your
task is missing under Task -> Actions -> Edit labels before importing.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from typing import Any, Optional

import cv2

import config
import store
from utils import (
    decode_label_map,
    imread_unicode,
    regions_from_annotation,
    regions_from_masks,
)

OUTPUT_XML = config.OUTPUT_DIR / "cvat_annotations.xml"


def _points_attr(polygon: list[dict[str, int]]) -> str:
    """CVAT polygon points: 'x1,y1;x2,y2;...' with one decimal of precision."""
    return ";".join(f"{p['x']:.1f},{p['y']:.1f}" for p in polygon)


def _build_meta(image_count: int) -> ET.Element:
    """Minimal <meta> block. CVAT reads labels from here when the file is used
    to seed a task; on upload to an existing task it matches by image name."""
    meta = ET.Element("meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "road-annotator-export"
    ET.SubElement(task, "size").text = str(image_count)
    ET.SubElement(task, "mode").text = "annotation"
    labels = ET.SubElement(task, "labels")
    for surface in config.SURFACE_CLASSES:
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = surface.key
        ET.SubElement(label, "type").text = "polygon"
        # CVAT wants BGR reversed into #RRGGBB.
        b, g, r = surface.color
        ET.SubElement(label, "color").text = f"#{r:02x}{g:02x}{b:02x}"
        ET.SubElement(label, "attributes")
    return meta


def build_export(scope: str = "annotated") -> tuple[Optional[ET.Element], dict[str, Any]]:
    """
    Assemble the CVAT annotations tree for every image in `scope`.

    Returns (root, summary). root is None when nothing qualified — an empty
    task file would import cleanly over the top of real work and silently
    wipe it, so the caller is expected to refuse rather than write one.
    """
    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = "1.1"

    processed: list[str] = []
    skipped: list[dict[str, str]] = []
    totals: dict[str, int] = {c.key: 0 for c in config.SURFACE_CLASSES}
    from_review = 0

    for frame_id, (image_path, document, origin) in enumerate(
        store.iter_exportable(scope)
    ):
        image = imread_unicode(image_path)
        if image is None:
            skipped.append({"name": image_path.name, "reason": "could not read image"})
            continue
        height, width = image.shape[:2]

        if origin == "human":
            # Straight out of the file, no re-approximation. This is the
            # fix for edits being silently discarded on export.
            regions = regions_from_annotation(document)
            from_review += 1
        else:
            label_map_path = store.label_map_path(image_path)
            label_map = imread_unicode(label_map_path, cv2.IMREAD_GRAYSCALE)
            if label_map is None:
                skipped.append({
                    "name": image_path.name,
                    "reason": f"no cached label map at {label_map_path.name} "
                              f"(run `python batch.py ./images` first)",
                })
                continue
            if label_map.shape != (height, width):
                skipped.append({
                    "name": image_path.name,
                    "reason": f"cached label map is {label_map.shape[1]}x"
                              f"{label_map.shape[0]}, image is {width}x{height} "
                              f"(re-run `python batch.py ./images`)",
                })
                continue
            # Already cleaned and ego-masked, so this is pure contour
            # extraction via the same code path the pipeline uses.
            regions, _ = regions_from_masks(decode_label_map(label_map))

        image_el = ET.SubElement(
            annotations, "image",
            id=str(frame_id), name=image_path.name,
            width=str(width), height=str(height),
        )
        counts: dict[str, int] = {c.key: 0 for c in config.SURFACE_CLASSES}
        for region in regions:
            if len(region.polygon) < 3:
                continue
            ET.SubElement(
                image_el, "polygon",
                label=region.class_key, source="manual", occluded="0", z_order="0",
                points=_points_attr(region.polygon),
            )
            counts[region.class_key] = counts.get(region.class_key, 0) + 1
            totals[region.class_key] = totals.get(region.class_key, 0) + 1

        processed.append(image_path.name)

    summary = {
        "scope": scope,
        "images": processed,
        "image_count": len(processed),
        "from_review": from_review,
        "from_label_map": len(processed) - from_review,
        "totals": totals,
        "skipped": skipped,
    }

    if not processed:
        return None, summary

    # meta is inserted right after <version> so the file matches CVAT's ordering.
    annotations.insert(1, _build_meta(len(processed)))
    ET.indent(annotations, space="  ")
    return annotations, summary


def write_export(scope: str = "annotated", destination=OUTPUT_XML) -> dict[str, Any]:
    """Build the export and write it to disk. Writes nothing if empty."""
    root, summary = build_export(scope)
    if root is None:
        return summary
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    summary["path"] = str(destination)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope", choices=("annotated", "approved"), default="annotated",
        help="which images to include (default: annotated)",
    )
    args = parser.parse_args()

    summary = write_export(args.scope)

    for entry in summary["skipped"]:
        print(f"  ! skip {entry['name']}: {entry['reason']}")

    if not summary["image_count"]:
        print(f"Nothing to export with scope '{args.scope}'.")
        if args.scope == "approved":
            print("  No images have been approved yet — approve them in the UI, "
                  "or use --scope annotated.")
        return

    for name in summary["images"]:
        print(f"  {name}")

    print(f"\nWrote {summary['path']}")
    print(f"  {summary['image_count']} images "
          f"({summary['from_review']} reviewed, "
          f"{summary['from_label_map']} straight from the model) | "
          + ", ".join(f"{v} {k}" for k, v in summary["totals"].items()))
    print(f"  labels used: {', '.join(repr(c.key) for c in config.SURFACE_CLASSES)} "
          f"(these must exist in your CVAT task)")


if __name__ == "__main__":
    main()
