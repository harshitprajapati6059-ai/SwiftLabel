"""
polygon.py - Mask cleaning and mask-to-polygon conversion, using OpenCV only.
Responsibility: binary mask in -> (cleaned mask, surface regions) out.

Everything here is per-class: the caller runs it once for road, once for
walkway, and so on. Nothing in this module knows which class it is
looking at, beyond carrying the label through onto the output regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class SurfaceRegion:
    """One exported polygon: a single connected patch of one surface class."""
    class_key: str
    class_id: int
    drivable: bool
    polygon: list[dict[str, int]]
    area_px: int
    bbox: tuple[int, int, int, int]   # x, y, w, h


def clean_mask(mask: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """
    Apply configurable morphological cleanup to a binary mask:
    opening (remove noise) -> closing (fill small gaps) -> hole filling
    -> small blob removal -> ego-vehicle exclusion.

    NOTE: ego-vehicle exclusion must run LAST. If it ran before hole
    filling, the hood cutout could be misread as an enclosed hole inside
    the surface region and filled straight back in, since a surface band
    can span the full image width and seal off everything beneath it.
    Running the exclusion after cleanup guarantees it can never be undone
    by a later step.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE)
    )

    cleaned = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel, iterations=config.MORPH_OPEN_ITERATIONS
    )
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE, kernel, iterations=config.MORPH_CLOSE_ITERATIONS
    )

    if config.FILL_HOLES_MAX_AREA_PX > 0:
        cleaned = _fill_small_holes(cleaned, config.FILL_HOLES_MAX_AREA_PX)

    cleaned = _remove_small_blobs(cleaned, config.MIN_BLOB_AREA_PX)

    cleaned = apply_ego_vehicle_mask(cleaned, image_bgr)

    return cleaned


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """
    Fill enclosed black holes inside white regions, but only those below
    max_area.

    Filling every hole regardless of size is wrong once more than one
    surface class exists: a green area legitimately encloses a flower bed
    or a footpath, and swallowing that cutout deletes a region another
    class is about to claim. Small holes are speckle from the model's
    boundary noise and should still go.
    """
    height, width = mask.shape
    background = cv2.bitwise_not(mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        background, connectivity=8
    )

    # A component touching the image border is outside the region, not a
    # hole in it.
    border_labels = set(labels[0, :]) | set(labels[-1, :]) \
        | set(labels[:, 0]) | set(labels[:, -1])

    filled = mask.copy()
    for label_id in range(1, num_labels):
        if label_id in border_labels:
            continue
        if stats[label_id, cv2.CC_STAT_AREA] <= max_area:
            filled[labels == label_id] = 255
    return filled


def _remove_small_blobs(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components smaller than min_area pixels."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)
    for label_id in range(1, num_labels):  # skip background (0)
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            output[labels == label_id] = 255
    return output


def merge_masks(
    masks: dict[str, np.ndarray], image_bgr: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """
    Collapse every class mask into ONE mask that becomes a single polygon.

    Union -> close -> fill all holes -> keep the largest component ->
    re-cut the ego vehicle.

    Returns the merged mask plus any notes worth surfacing as warnings.

    Two different gaps have to be closed, which is why both a closing
    pass and a hole fill are needed:

      * a car in the middle of the road leaves an ENCLOSED hole, which
        only the hole fill reaches;
      * a car parked against the kerb bites a notch that is open to the
        outside, which the hole fill cannot see and only the closing
        kernel can bridge.

    Re-applying the ego mask at the end is not optional. Closing with a
    kernel this wide will happily bridge straight across the vehicle
    hood, and filling all holes would then seal it in — undoing the one
    exclusion the whole frame depends on.
    """
    notes: list[str] = []
    shape = next(iter(masks.values())).shape

    union = np.zeros(shape, dtype=np.uint8)
    for mask in masks.values():
        union = cv2.bitwise_or(union, mask)

    if config.MERGE_CLOSE_KERNEL_PX > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.MERGE_CLOSE_KERNEL_PX, config.MERGE_CLOSE_KERNEL_PX),
        )
        union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel)

    if config.MERGE_FILL_ALL_HOLES:
        # No area bound here: an enclosed hole inside the merged region
        # is by definition something standing on the surface, and the
        # brief asks for those to be covered rather than cut around.
        union = _fill_small_holes(union, max_area=union.size)

    # "One polygon" has to mean one component. Anything genuinely
    # disconnected after that closing pass is islanded — reported rather
    # than dropped in silence.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
    if num_labels > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(np.argmax(areas)) + 1
        discarded = int(areas.sum() - areas[largest - 1])
        union = np.where(labels == largest, 255, 0).astype(np.uint8)
        notes.append(
            f"{num_labels - 2} disconnected region(s) ({discarded}px) dropped to "
            f"keep a single polygon — raise MERGE_CLOSE_KERNEL_PX to absorb them."
        )

    union = apply_ego_vehicle_mask(union, image_bgr)

    return union, notes


def mask_to_regions(
    mask: np.ndarray, surface: config.SurfaceClass
) -> tuple[list[SurfaceRegion], int]:
    """
    Convert a cleaned binary mask into one simplified polygon per
    connected region of that class.

    Returns:
        regions: every region above the area floor, largest first.
        dropped: how many contours were discarded for being too small
                 (reported so a silently vanishing region is visible).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    significant = [c for c in contours if cv2.contourArea(c) >= config.MIN_CONTOUR_AREA_PX]
    dropped = len(contours) - len(significant)

    significant.sort(key=cv2.contourArea, reverse=True)
    if config.MAX_REGIONS_PER_CLASS > 0:
        dropped += max(0, len(significant) - config.MAX_REGIONS_PER_CLASS)
        significant = significant[: config.MAX_REGIONS_PER_CLASS]

    regions: list[SurfaceRegion] = []
    for contour in significant:
        perimeter = cv2.arcLength(contour, True)
        epsilon = config.POLYGON_EPSILON_RATIO * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # approxPolyDP can collapse a thin sliver below three points,
        # which is no longer a polygon anything downstream can import.
        if len(approx) < 3:
            dropped += 1
            continue

        regions.append(SurfaceRegion(
            class_key=surface.key,
            class_id=surface.class_id,
            drivable=surface.drivable,
            polygon=[{"x": int(pt[0][0]), "y": int(pt[0][1])} for pt in approx],
            area_px=int(cv2.contourArea(contour)),
            bbox=tuple(int(v) for v in cv2.boundingRect(contour)),
        ))

    return regions, dropped


def apply_ego_vehicle_mask(mask: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """
    Exclude the vehicle hood/mount from a surface mask.

    The hardcoded polygon defines a SEARCH ZONE only (scaled to the
    current image size, then dilated by a safety margin). Inside that
    zone, we detect actual near-black pixels fresh for this specific
    image and exclude only those — this handles the hood's boundary
    shifting a few pixels frame to frame, while never touching real
    shadows outside the known zone.
    """
    if not config.EGO_VEHICLE_MASK_ENABLED:
        return mask

    h, w = mask.shape
    ref_w, ref_h = config.EGO_VEHICLE_MASK_REFERENCE_SIZE
    scale_x, scale_y = w / ref_w, h / ref_h

    polygon = np.array(
        [[int(x * scale_x), int(y * scale_y)] for x, y in config.EGO_VEHICLE_MASK_POLYGON],
        dtype=np.int32,
    )

    # Build the search zone: filled polygon + margin via dilation
    search_zone = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(search_zone, [polygon], 255)
    if config.EGO_VEHICLE_SEARCH_MARGIN_PX > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.EGO_VEHICLE_SEARCH_MARGIN_PX, config.EGO_VEHICLE_SEARCH_MARGIN_PX),
        )
        search_zone = cv2.dilate(search_zone, kernel)

    # Find actual black pixels, but only trust that finding inside the search zone
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, black_pixels = cv2.threshold(
        gray, config.EGO_VEHICLE_BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV
    )
    hood_mask = cv2.bitwise_and(black_pixels, search_zone)

    result = mask.copy()
    result[hood_mask == 255] = 0
    return result
