"""
store.py - Everything that reads or writes annotation state on disk.

The pipeline owns output/polygons/ and never reads it back. This module
owns output/edited/, which holds the HUMAN copy of an annotation, and is
the only place that decides what state an image is in.

Two files per image, and the split is the whole design:

    output/polygons/<stem>_polygon.json    what the model produced
    output/edited/<stem>_polygon.json      what the human decided

The model's output is never overwritten, so "revert to AI" is just
deleting the second file, and a bad review can always be thrown away
without re-running inference.

Approval lives in the human file rather than being inferred from it.
An image the model got right is approved with no edits at all, so
"a human file exists" and "a human has checked this" are different
facts — collapsing them would make a perfect prediction indistinguishable
from one nobody has looked at yet.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import numpy as np

import config
from utils import imread_unicode, is_supported_image, load_json, logger, save_json

# Longest edge of a gallery thumbnail, in pixels.
THUMBNAIL_MAX_EDGE = 240
THUMBNAIL_JPEG_QUALITY = 82

# Image states, in workflow order. Also the values the API reports.
UNANNOTATED = "unannotated"
AUTO = "auto"
EDITED = "edited"
APPROVED = "approved"


class ImageNotFound(Exception):
    """Requested image is not present in the images directory."""


class UnsafeImageName(Exception):
    """Requested name escapes the images directory, or is not a plain filename."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def resolve_image(name: str) -> Path:
    """
    Turn a client-supplied filename into a real path inside IMAGES_DIR.

    Every route takes an image name straight from the URL, so this is the
    one place that has to be paranoid. A name is accepted only if it has
    no directory component at all and still resolves to a direct child of
    IMAGES_DIR — which rejects "../config.py", absolute paths, and (on
    Windows) alternate spellings like "..\\config.py" that a naive
    startswith check on the raw string would let through.
    """
    if not name or name in (".", ".."):
        raise UnsafeImageName(name)

    candidate = Path(name)
    if candidate.is_absolute() or len(candidate.parts) != 1:
        raise UnsafeImageName(name)

    images_dir = config.IMAGES_DIR.resolve()
    resolved = (images_dir / candidate.name).resolve()
    if resolved.parent != images_dir:
        raise UnsafeImageName(name)

    if not resolved.is_file():
        raise ImageNotFound(name)
    if not is_supported_image(resolved):
        raise ImageNotFound(name)

    return resolved


def auto_annotation_path(image_path: Path) -> Path:
    """Where the pipeline writes this image's polygons."""
    return config.POLYGONS_DIR / f"{image_path.stem}_polygon.json"


def human_annotation_path(image_path: Path) -> Path:
    """Where the review UI writes this image's polygons."""
    return config.EDITED_DIR / f"{image_path.stem}_polygon.json"


def label_map_path(image_path: Path) -> Path:
    """The cached label map the CVAT export falls back to."""
    return config.MASKS_DIR / f"{image_path.stem}_labels.png"


def list_images() -> list[Path]:
    """Every supported image in IMAGES_DIR, in stable display order."""
    if not config.IMAGES_DIR.is_dir():
        return []
    return sorted(
        (p for p in config.IMAGES_DIR.iterdir() if p.is_file() and is_supported_image(p)),
        key=lambda p: p.name.lower(),
    )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_annotation(image_path: Path) -> tuple[Optional[dict[str, Any]], str]:
    """
    Return (document, origin) for an image, preferring the human copy.

    origin is "human", "auto", or "none" — the caller needs to know which
    it got, because saving a human copy over the model's output is the
    normal case but reverting is not.
    """
    human = load_json(human_annotation_path(image_path))
    if human is not None:
        return human, "human"

    auto = load_json(auto_annotation_path(image_path))
    if auto is not None:
        return auto, "auto"

    return None, "none"


def _shape_points(document: dict[str, Any]) -> list[list[tuple[int, int]]]:
    """Geometry only, rounded to whole pixels, for comparing two documents."""
    return [
        [(int(round(float(p["x"]))), int(round(float(p["y"]))))
         for p in shape.get("points", [])]
        for shape in document.get("shapes", [])
    ]


def geometry_changed(image_path: Path, document: dict[str, Any]) -> bool:
    """
    Has this document's geometry actually diverged from the model's output?

    Decided here rather than trusting a client-supplied "edited" flag: the
    UI saves on every navigation, so it would otherwise mark images as
    hand-corrected just for having been opened.
    """
    auto = load_json(auto_annotation_path(image_path))
    if auto is None:
        # Nothing to compare against — a hand-drawn image on a frame that
        # was never run through the model is manual by definition.
        return True
    return _shape_points(auto) != _shape_points(document)


def state(image_path: Path) -> str:
    """Which of the four workflow states this image is in."""
    human = load_json(human_annotation_path(image_path))
    if human is not None:
        if human.get("approved"):
            return APPROVED
        if human.get("source") == "manual":
            return EDITED
        # Saved but unchanged and unapproved — no more reviewed than an
        # image nobody has opened, so it reports as auto.
        return AUTO

    if auto_annotation_path(image_path).exists():
        return AUTO
    return UNANNOTATED


def summarise(image_path: Path) -> dict[str, Any]:
    """The per-image record the gallery and filmstrip are built from."""
    document, origin = read_annotation(image_path)
    stat = image_path.stat()

    shapes = document.get("shapes", []) if document else []
    return {
        "name": image_path.name,
        "state": state(image_path),
        "origin": origin,
        "shape_count": len(shapes),
        "point_count": sum(len(s.get("points", [])) for s in shapes),
        "width": document.get("image_width") if document else None,
        "height": document.get("image_height") if document else None,
        "size_bytes": stat.st_size,
        "modified": stat.st_mtime,
        "approved_at": document.get("approved_at") if document else None,
        "warnings": _warnings_for(image_path),
    }


def _warnings_for(image_path: Path) -> list[str]:
    """Pipeline warnings from the last run, so the gallery can flag them."""
    metadata = load_json(config.POLYGONS_DIR / f"{image_path.stem}_metadata.json")
    return list(metadata.get("warnings", [])) if metadata else []


def summarise_all() -> list[dict[str, Any]]:
    return [summarise(p) for p in list_images()]


def counts_by_state() -> dict[str, int]:
    """State histogram, for the header counter and the export dialog."""
    tally = {UNANNOTATED: 0, AUTO: 0, EDITED: 0, APPROVED: 0}
    for image_path in list_images():
        tally[state(image_path)] += 1
    return tally


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalise_shapes(
    shapes: list[dict[str, Any]], width: int, height: int
) -> list[dict[str, Any]]:
    """
    Coerce client-supplied shapes into the on-disk schema.

    Points come back from a canvas as floats; they are rounded to whole
    pixels here so the saved file matches what build_annotation() writes
    and the CVAT export cannot drift by a subpixel between the two paths.

    They are also clamped to the frame. Dragging a vertex past the edge of
    the image is an easy accident at high zoom, and CVAT rejects a polygon
    with coordinates outside the declared frame size — so an off-canvas
    vertex would fail the whole import, long after the mistake was made.
    """
    cleaned: list[dict[str, Any]] = []

    for index, shape in enumerate(shapes):
        points = [
            {
                "x": min(max(int(round(float(p["x"]))), 0), width - 1),
                "y": min(max(int(round(float(p["y"]))), 0), height - 1),
            }
            for p in shape.get("points", [])
        ]
        if len(points) < 3:
            continue

        key = shape.get("label", "")
        surface = config.SURFACE_CLASSES_BY_KEY.get(key)
        if surface is None:
            # An unknown label would be rejected by CVAT on import and is
            # far more likely to be a stale file than a deliberate choice.
            surface = config.SURFACE_CLASSES[0]
            logger.warning(f"Unknown label {key!r}; falling back to {surface.key!r}")

        # Same two OpenCV calls the pipeline uses in mask_to_regions, so an
        # edited shape reports its geometry on exactly the same basis as a
        # generated one rather than by a subtly different convention.
        contour = np.array([[p["x"], p["y"]] for p in points], dtype=np.int32)
        cleaned.append({
            "id": index,
            "class_id": surface.class_id,
            "label": surface.key,
            "drivable": surface.drivable,
            "shape_type": "polygon",
            "points": points,
            "bbox": [int(v) for v in cv2.boundingRect(contour)],
            "area_px": int(abs(cv2.contourArea(contour))),
        })

    return cleaned


def save_annotation(
    image_path: Path,
    shapes: list[dict[str, Any]],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> dict[str, Any]:
    """
    Write the human copy of an annotation.

    Saving does NOT approve — approval is a separate deliberate action, and
    the UI autosaves on navigation. It also does not clear an existing
    approval unless the geometry actually changed, so re-opening an
    approved image and moving on leaves it approved.
    """
    existing = load_json(human_annotation_path(image_path)) or {}
    previous_points = _shape_points(existing) if existing else None

    document = _base_document(image_path, width, height)
    document["shapes"] = _normalise_shapes(
        shapes, document["image_width"], document["image_height"]
    )

    manual = geometry_changed(image_path, document)
    document["source"] = "manual" if manual else "auto"
    document["edited_at"] = _now()

    was_approved = bool(existing.get("approved"))
    changed_since_approval = (
        previous_points is not None and previous_points != _shape_points(document)
    )
    keep_approval = was_approved and not changed_since_approval

    document["approved"] = keep_approval
    document["approved_at"] = existing.get("approved_at") if keep_approval else None

    save_json(document, human_annotation_path(image_path))
    return document


def _base_document(
    image_path: Path, width: Optional[int], height: Optional[int]
) -> dict[str, Any]:
    """
    Start a human document, reusing the dimensions already on record.

    Falls back to decoding the image only when neither the client nor an
    existing annotation supplied them — the dimensions must be right or
    CVAT will place every polygon against the wrong frame size.
    """
    known, _ = read_annotation(image_path)
    resolved_width = width or (known or {}).get("image_width")
    resolved_height = height or (known or {}).get("image_height")

    if not resolved_width or not resolved_height:
        image = imread_unicode(image_path)
        if image is None:
            raise ValueError(f"Could not read image dimensions: {image_path.name}")
        resolved_height, resolved_width = image.shape[:2]

    return {
        "image_name": image_path.name,
        "image_width": int(resolved_width),
        "image_height": int(resolved_height),
        "classes": [
            {"class_id": c.class_id, "label": c.key, "drivable": c.drivable}
            for c in config.SURFACE_CLASSES
        ],
        "shapes": [],
    }


def approve(image_path: Path) -> dict[str, Any]:
    """
    Mark an image reviewed and signed off.

    If no human copy exists yet the model's output is promoted into one
    first — approving a prediction you did not need to touch is the
    common case, and it still has to produce a human-owned file so the
    export knows the annotation was actually looked at.
    """
    document = load_json(human_annotation_path(image_path))

    if document is None:
        auto = load_json(auto_annotation_path(image_path))
        if auto is None:
            raise ValueError(
                f"{image_path.name} has no annotation to approve — "
                f"run auto-annotate or draw one first."
            )
        document = dict(auto)
        document["source"] = "auto"
        document["edited_at"] = None

    document["approved"] = True
    document["approved_at"] = _now()
    save_json(document, human_annotation_path(image_path))
    return document


def unapprove(image_path: Path) -> Optional[dict[str, Any]]:
    """Withdraw approval, keeping any edits."""
    document = load_json(human_annotation_path(image_path))
    if document is None:
        return None
    document["approved"] = False
    document["approved_at"] = None
    save_json(document, human_annotation_path(image_path))
    return document


def revert(image_path: Path) -> bool:
    """
    Discard the human copy entirely, returning the image to the model's
    output. Clears edits and approval together — a clean restart.
    """
    path = human_annotation_path(image_path)
    if not path.exists():
        return False
    path.unlink()
    return True


def delete_image(image_path: Path) -> None:
    """Remove an image and every derived artefact belonging to it."""
    stem = image_path.stem
    derived = [
        human_annotation_path(image_path),
        auto_annotation_path(image_path),
        config.POLYGONS_DIR / f"{stem}_metadata.json",
        config.MASKS_DIR / f"{stem}_mask.png",
        config.MASKS_DIR / f"{stem}_labels.png",
        config.OVERLAYS_DIR / f"{stem}_overlay.png",
        config.THUMBS_DIR / f"{stem}.jpg",
    ]
    for path in derived:
        path.unlink(missing_ok=True)
    image_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------

def thumbnail(image_path: Path) -> Optional[Path]:
    """
    Path to a cached gallery thumbnail, generating it if needed.

    Regenerated when the source is newer than the cache, so replacing an
    image under the same name does not leave a stale tile behind.
    """
    cached = config.THUMBS_DIR / f"{image_path.stem}.jpg"
    if cached.exists() and cached.stat().st_mtime >= image_path.stat().st_mtime:
        return cached

    image = imread_unicode(image_path)
    if image is None:
        return None

    height, width = image.shape[:2]
    scale = THUMBNAIL_MAX_EDGE / float(max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    ok, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), THUMBNAIL_JPEG_QUALITY]
    )
    if not ok:
        return None

    config.THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    buffer.tofile(str(cached))
    return cached


def iter_exportable(scope: str) -> Iterator[tuple[Path, dict[str, Any], str]]:
    """
    Yield (image_path, document, origin) for every image the given export
    scope should include.

    "approved"  - only images signed off by a human.
    "annotated" - anything carrying an annotation at all, reviewed or not.
    """
    if scope not in ("approved", "annotated"):
        raise ValueError(f"Unknown export scope: {scope!r}")

    for image_path in list_images():
        current = state(image_path)
        if current == UNANNOTATED:
            continue
        if scope == "approved" and current != APPROVED:
            continue

        document, origin = read_annotation(image_path)
        if document is None:
            continue
        yield image_path, document, origin
