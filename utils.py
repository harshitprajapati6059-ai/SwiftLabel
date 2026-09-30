"""
utils.py - Shared helpers: logging, IO validation, JSON saving, and the
single orchestrator function that wires segment -> polygon -> preview
together. Both test.py and batch.py call run_pipeline() so the pipeline
logic only exists in one place.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import numpy as np

import config

# A Windows console defaults to cp1252, which cannot encode the tick marks
# and dashes used in reports and warnings — printing one raises
# UnicodeEncodeError and kills the run *after* the images were already
# written. Both entry points import this module, so forcing UTF-8 here
# covers print() and the logger's console handler in one place.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
from polygon import SurfaceRegion, clean_mask, mask_to_regions, merge_masks
from preview import build_overlay
from segment import SurfaceSegmenter


def get_logger(name: str = "road_annotator") -> logging.Logger:
    """Return a logger that writes to both console and logs/pipeline.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


logger = get_logger()


def is_supported_image(path: Path) -> bool:
    """Return True if the file extension is one we know how to process."""
    return path.suffix.lower() in config.SUPPORTED_EXTENSIONS


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    cv2.imread that works with non-ASCII paths on Windows.

    OpenCV's imread passes the filename to the C runtime as a byte string,
    so on Windows a path containing anything outside the active ANSI code
    page simply fails — and it fails by returning None, which is
    indistinguishable from a corrupt file. Once images arrive by upload
    rather than being copied in by hand, a name like "straße.jpg" is
    entirely ordinary, so the bytes are read in Python and decoded from
    memory instead.
    """
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, flags)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """cv2.imwrite counterpart to imread_unicode — same Windows limitation."""
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(str(path))
    return True


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk. Raises ValueError with a clear message on failure."""
    if not path.exists():
        raise ValueError(f"Image not found: {path}")
    if not is_supported_image(path):
        raise ValueError(f"Unsupported file type: {path.suffix}")

    image = imread_unicode(path)
    if image is None:
        raise ValueError(f"Could not decode image (corrupt or unreadable): {path}")
    return image


def save_json(data: dict[str, Any], path: Path) -> None:
    """
    Write a dict to disk as pretty-printed JSON, atomically.

    The review UI saves on every navigation, so a crash or a killed server
    mid-write is a realistic way to lose work. Writing to a temp file in
    the same directory and then os.replace()-ing it means a reader either
    sees the previous complete file or the new complete one, never a
    half-written one. Same directory matters: replace() is only atomic
    within a filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_json(path: Path) -> Optional[dict[str, Any]]:
    """Read a JSON document, returning None if it is absent or unreadable."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Ignoring unreadable JSON at {path}: {exc}")
        return None


@contextmanager
def config_overrides(**values: Any) -> Iterator[None]:
    """
    Temporarily rebind module-level values in config.py, restoring them on
    exit.

    The pipeline reads config.* at call time rather than taking a settings
    object, so this is what lets the UI re-run one image with a different
    merge kernel without editing config.py or refactoring four modules.

    SAFE ONLY BECAUSE INFERENCE IS SERIALISED. jobs.py runs a single
    worker thread, so no two runs are ever inside this block at once. If a
    second worker is ever added, concurrent runs would see each other's
    values and this must become a real settings parameter instead.

    Unknown names raise rather than silently creating a new attribute that
    nothing reads — a typo'd override should be loud, not a no-op.
    """
    unknown = [k for k in values if not hasattr(config, k)]
    if unknown:
        raise AttributeError(f"config has no such setting(s): {', '.join(sorted(unknown))}")

    previous = {k: getattr(config, k) for k in values}
    for key, value in values.items():
        setattr(config, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(config, key, value)


@dataclass
class PipelineResult:
    """Everything the caller (test.py / batch.py) needs after processing one image."""
    image_name: str
    processing_time_sec: float
    model_used: str
    region_counts: dict[str, int]          # class key -> polygons exported
    coverage_pct: dict[str, float]         # class key -> % of frame covered
    confidences: dict[str, float]          # class key -> mean softmax probability
    drivable_area_pct: float
    timestamp: str
    mask_path: str
    label_map_path: str
    overlay_path: str
    polygon_path: str
    metadata_path: str
    warnings: list[str] = field(default_factory=list)

    @property
    def total_regions(self) -> int:
        return sum(self.region_counts.values())


def build_annotation(
    image_path: Path,
    image_shape: tuple[int, ...],
    regions: list[SurfaceRegion],
) -> dict[str, Any]:
    """
    Assemble the exported annotation document.

    One entry per surface region, carrying the class label, its stable
    class_id from config, and the `drivable` flag — the actual ADAS
    question — so a consumer never has to re-derive drivability by string
    matching on the label.
    """
    height, width = image_shape[:2]

    shapes = [
        {
            "id": index,
            "class_id": region.class_id,
            "label": region.class_key,
            "drivable": region.drivable,
            "shape_type": "polygon",
            "points": region.polygon,
            "bbox": list(region.bbox),
            "area_px": region.area_px,
        }
        for index, region in enumerate(regions)
    ]

    return {
        "image_name": image_path.name,
        "image_width": int(width),
        "image_height": int(height),
        "classes": [
            {"class_id": c.class_id, "label": c.key, "drivable": c.drivable}
            for c in config.SURFACE_CLASSES
        ],
        "shapes": shapes,
    }


_segmenter: Optional[SurfaceSegmenter] = None


def get_segmenter() -> SurfaceSegmenter:
    """Lazily instantiate the model once and reuse it across images."""
    global _segmenter
    if _segmenter is None:
        _segmenter = SurfaceSegmenter()
    return _segmenter


def resolve_overlaps(
    masks: dict[str, np.ndarray], label_map: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Force the per-class masks to be mutually exclusive.

    The model's own output is exclusive (one argmax per pixel), but each
    class is then cleaned independently, and morphological closing grows
    every class outward by roughly half a kernel. Along a kerb line that
    means road and walkway both end up claiming the same few pixels.

    Two things go wrong if that is left alone. A pixel gets exported as
    both drivable and not drivable, which is the one question this whole
    pipeline exists to answer. And the cached label map — a single value
    per pixel — has to pick a winner, so the polygons export_cvat.py
    rebuilds stop matching the overlay a human reviewed.

    Contested pixels are given to whichever class actually won the
    argmax there, so cleaning can only ever fill gaps, never repaint a
    boundary. Where the raw prediction was unlabelled (or its class lost
    the pixel to blob removal or the ego mask), declaration order in
    config.SURFACE_CLASSES breaks the tie.
    """
    from segment import UNLABELLED

    combined = np.full(label_map.shape, UNLABELLED, dtype=np.int16)

    # Pass 1: declaration order, first-listed wins — so assign in reverse.
    for surface in reversed(config.SURFACE_CLASSES):
        mask = masks.get(surface.key)
        if mask is not None:
            combined[mask == 255] = surface.class_id

    # Pass 2: the raw prediction overrides. `& (mask == 255)` matters —
    # it stops a pixel that cleaning deliberately removed (small blob,
    # vehicle hood) from being resurrected by its original label.
    for surface in config.SURFACE_CLASSES:
        mask = masks.get(surface.key)
        if mask is None:
            continue
        combined[(label_map == surface.class_id) & (mask == 255)] = surface.class_id

    resolved: dict[str, np.ndarray] = {}
    for surface in config.SURFACE_CLASSES:
        exclusive = np.zeros(label_map.shape, dtype=np.uint8)
        exclusive[combined == surface.class_id] = 255
        if exclusive.any():
            resolved[surface.key] = exclusive
    return resolved


def encode_label_map(masks: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """
    Pack the per-class cleaned masks into one single-channel image, with
    pixel value = class_id + 1 and 0 meaning unlabelled.

    Written to disk so export_cvat.py can rebuild every polygon offline
    from a previous run without re-invoking the model. The +1 offset
    exists because road is class_id 0 and would otherwise be
    indistinguishable from background.
    """
    label_map = np.zeros(shape, dtype=np.uint8)
    for surface in config.SURFACE_CLASSES:
        mask = masks.get(surface.key)
        if mask is not None:
            label_map[mask == 255] = surface.class_id + 1
    return label_map


def decode_label_map(label_map: np.ndarray) -> dict[str, np.ndarray]:
    """Inverse of encode_label_map: single-channel image -> per-class masks."""
    masks: dict[str, np.ndarray] = {}
    for surface in config.SURFACE_CLASSES:
        mask = np.zeros(label_map.shape, dtype=np.uint8)
        mask[label_map == surface.class_id + 1] = 255
        if mask.any():
            masks[surface.key] = mask
    return masks


def colorize_label_map(masks: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Render the per-class masks as a flat colour image for quick eyeballing."""
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    for surface in config.SURFACE_CLASSES:
        mask = masks.get(surface.key)
        if mask is not None:
            canvas[mask == 255] = surface.color
    return canvas


def regions_from_masks(masks: dict[str, np.ndarray]) -> tuple[list[SurfaceRegion], list[str]]:
    """
    Turn cleaned per-class masks into exported polygons.

    Shared by run_pipeline and export_cvat so the two can never drift
    into producing different polygons from the same mask.
    """
    regions: list[SurfaceRegion] = []
    notes: list[str] = []
    for surface in config.SURFACE_CLASSES:
        mask = masks.get(surface.key)
        if mask is None or not mask.any():
            continue
        class_regions, dropped = mask_to_regions(mask, surface)
        regions.extend(class_regions)
        if dropped:
            notes.append(
                f"{surface.key}: {dropped} region(s) below "
                f"{config.MIN_CONTOUR_AREA_PX}px dropped."
            )
    return regions, notes


def regions_from_annotation(document: dict[str, Any]) -> list[SurfaceRegion]:
    """
    Rebuild SurfaceRegion objects from a saved annotation document — the
    inverse of build_annotation().

    This is what lets the CVAT export emit a hand-corrected polygon exactly
    as it was drawn. The alternative, re-deriving it from the cached label
    map, would run the contour approximation again and shift every vertex
    the reviewer placed by hand.

    bbox and area_px are recomputed from the points rather than trusted
    from the file: after a vertex drag the stored values still describe the
    shape as it was before the edit.
    """
    regions: list[SurfaceRegion] = []

    for shape in document.get("shapes", []):
        points = [
            {"x": int(round(float(p["x"]))), "y": int(round(float(p["y"])))}
            for p in shape.get("points", [])
        ]
        # approxPolyDP guards against this upstream, but a hand-drawn shape
        # can be abandoned mid-draw and a 2-point "polygon" is not
        # importable anywhere.
        if len(points) < 3:
            continue

        # Prefer what the document itself recorded; fall back to the class
        # table only if the shape predates those fields.
        surface = config.SURFACE_CLASSES_BY_KEY.get(shape.get("label", ""))
        contour = np.array([[p["x"], p["y"]] for p in points], dtype=np.int32)

        regions.append(SurfaceRegion(
            class_key=shape.get("label") or (surface.key if surface else "unknown"),
            class_id=int(shape.get("class_id", surface.class_id if surface else 0)),
            drivable=bool(shape.get("drivable", surface.drivable if surface else False)),
            polygon=points,
            area_px=int(abs(cv2.contourArea(contour))),
            bbox=tuple(int(v) for v in cv2.boundingRect(contour)),
        ))

    return regions


def run_pipeline(
    image_path: Path, overrides: Optional[dict[str, Any]] = None
) -> PipelineResult:
    """
    Run the full pipeline on a single image:
    load -> segment -> clean per class -> polygons -> preview -> save.

    `overrides` temporarily rebinds config values for this run only (see
    config_overrides). The values used are recorded into the metadata, so a
    polygon always carries the settings that produced it.
    """
    with config_overrides(**(overrides or {})):
        return _run_pipeline(image_path, overrides or {})


def _run_pipeline(image_path: Path, overrides: dict[str, Any]) -> PipelineResult:
    warnings: list[str] = []
    start = time.time()

    image = load_image(image_path)
    segmenter = get_segmenter()

    label_map, confidences = segmenter.predict(image)
    frame_shape = label_map.shape

    if segmenter.unsupported_classes:
        warnings.append(
            f"{', '.join(segmenter.unsupported_classes)} not produced by "
            f"'{config.MODEL_NAME}' — assign by hand during review."
        )

    # Clean each class independently. Sharing one pass would let the
    # morphology of a large class eat into a small neighbouring one.
    masks: dict[str, np.ndarray] = {}
    for surface in config.SURFACE_CLASSES:
        raw = np.zeros(frame_shape, dtype=np.uint8)
        raw[label_map == surface.class_id] = 255
        if not raw.any():
            continue
        cleaned = clean_mask(raw, image)
        if cleaned.any():
            masks[surface.key] = cleaned

    if not masks:
        raise ValueError("No drivable-surface classes detected in this image.")

    # Must run before anything reads the masks, so the overlay, the JSON
    # and the cached label map all describe the identical partition.
    masks = resolve_overlaps(masks, label_map)

    if config.MERGE_INTO_SINGLE_POLYGON:
        merged, merge_notes = merge_masks(masks, image)
        warnings.extend(merge_notes)
        # The merged region is attributed to the first declared class,
        # which with a single-class table is the only one there is.
        masks = {config.SURFACE_CLASSES[0].key: merged}

    regions, notes = regions_from_masks(masks)
    warnings.extend(notes)

    if not regions:
        raise ValueError("Surfaces were detected but no region was large enough to export.")

    region_counts = {c.key: 0 for c in config.SURFACE_CLASSES}
    for region in regions:
        region_counts[region.class_key] += 1

    total_px = float(frame_shape[0] * frame_shape[1])
    coverage_pct = {
        key: round(100.0 * float((mask == 255).sum()) / total_px, 2)
        for key, mask in masks.items()
    }
    drivable_area_pct = round(sum(
        pct for key, pct in coverage_pct.items()
        if config.SURFACE_CLASSES_BY_KEY[key].drivable
    ), 2)

    for key, confidence in confidences.items():
        if confidence < config.CONFIDENCE_THRESHOLD:
            warnings.append(
                f"Low confidence for '{key}' ({confidence:.2f}) — inspect the overlay."
            )
    for surface in config.SURFACE_CLASSES:
        if surface.key not in segmenter.unsupported_classes and not region_counts[surface.key]:
            warnings.append(f"No '{surface.key}' region found — verify against the overlay.")

    overlay = build_overlay(image, masks, regions)

    stem = image_path.stem
    mask_path = config.MASKS_DIR / f"{stem}_mask.png"
    label_map_path = config.MASKS_DIR / f"{stem}_labels.png"
    overlay_path = config.OVERLAYS_DIR / f"{stem}_overlay.png"
    polygon_path = config.POLYGONS_DIR / f"{stem}_polygon.json"
    metadata_path = config.POLYGONS_DIR / f"{stem}_metadata.json"

    imwrite_unicode(mask_path, colorize_label_map(masks, frame_shape))
    imwrite_unicode(label_map_path, encode_label_map(masks, frame_shape))
    imwrite_unicode(overlay_path, overlay)
    save_json(build_annotation(image_path, image.shape, regions), polygon_path)

    elapsed = round(time.time() - start, 3)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    metadata = {
        "image_name": image_path.name,
        "processing_time_sec": elapsed,
        "model_used": config.MODEL_NAME,
        "region_counts": region_counts,
        "coverage_percentage": coverage_pct,
        "drivable_area_percentage": drivable_area_pct,
        "confidence": {k: round(v, 4) for k, v in confidences.items()},
        "regions": [
            {
                "label": r.class_key,
                "class_id": r.class_id,
                "drivable": r.drivable,
                "point_count": len(r.polygon),
                "area_px": r.area_px,
                "bbox": list(r.bbox),
            }
            for r in regions
        ],
        "timestamp": timestamp,
        # Empty for a default run. Recorded so a polygon produced with a
        # one-off merge kernel is not mistaken later for one produced with
        # the values currently sitting in config.py.
        "overrides": overrides,
        "warnings": warnings,
    }
    save_json(metadata, metadata_path)

    for w in warnings:
        logger.warning(f"{image_path.name}: {w}")
    counts = " ".join(f"{k}={v}" for k, v in region_counts.items())
    logger.info(
        f"{image_path.name} | {elapsed}s | {counts} | drivable={drivable_area_pct}%"
    )

    return PipelineResult(
        image_name=image_path.name,
        processing_time_sec=elapsed,
        model_used=config.MODEL_NAME,
        region_counts=region_counts,
        coverage_pct=coverage_pct,
        confidences=confidences,
        drivable_area_pct=drivable_area_pct,
        timestamp=timestamp,
        mask_path=str(mask_path),
        label_map_path=str(label_map_path),
        overlay_path=str(overlay_path),
        polygon_path=str(polygon_path),
        metadata_path=str(metadata_path),
        warnings=warnings,
    )
