"""
preview.py - Draws the human-inspectable overlay image.
Responsibility: image + per-class masks + regions -> annotated preview.
No file saving here (the caller decides where to write it).

One colour per surface class (defined in config.SURFACE_CLASSES), not
one colour per instance. The reviewer's question is "is this patch the
right CLASS", so neighbouring patches of the same class sharing a colour
is correct — the outlines still separate them.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

import config
from polygon import SurfaceRegion


def build_overlay(
    image_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    regions: Sequence[SurfaceRegion],
) -> np.ndarray:
    """
    Compose the original image with a translucent per-class fill, a solid
    outline per exported polygon, and a colour legend.
    """
    base = image_bgr.copy()

    colored = np.zeros_like(base)
    painted = np.zeros(base.shape[:2], dtype=bool)

    for surface in config.SURFACE_CLASSES:
        mask = masks.get(surface.key)
        if mask is None:
            continue
        selected = mask == 255
        colored[selected] = surface.color
        painted |= selected

    overlay = base.copy()
    # Blend only where something was actually painted, so unlabelled
    # areas keep their original brightness instead of being dimmed
    # toward black by an all-zero colour layer.
    overlay[painted] = cv2.addWeighted(
        colored, config.OVERLAY_FILL_ALPHA,
        base, 1 - config.OVERLAY_FILL_ALPHA, 0
    )[painted]

    for region in regions:
        surface = config.SURFACE_CLASSES_BY_KEY[region.class_key]
        pts = np.array([[p["x"], p["y"]] for p in region.polygon], dtype=np.int32)
        cv2.polylines(
            overlay, [pts], isClosed=True,
            color=surface.color, thickness=config.OVERLAY_OUTLINE_THICKNESS,
            lineType=cv2.LINE_AA,
        )
        if config.OVERLAY_DRAW_VERTICES:
            for p in region.polygon:
                cv2.circle(overlay, (p["x"], p["y"]), config.OVERLAY_VERTEX_RADIUS,
                           config.TEXT_COLOR, -1)

    if config.OVERLAY_DRAW_LEGEND:
        _draw_legend(overlay, regions)

    return overlay


def _draw_legend(overlay: np.ndarray, regions: Sequence[SurfaceRegion]) -> None:
    """
    List each class, its colour and how many polygons it produced.

    Classes with zero regions are still listed, greyed out: "walkway: 0"
    is a review signal, whereas an absent row just looks like the legend
    was short that frame.
    """
    counts: dict[str, int] = {c.key: 0 for c in config.SURFACE_CLASSES}
    for region in regions:
        counts[region.class_key] += 1

    x, y = 10, 24
    line_height = 22

    # Backdrop: an overcast sky is near-white and a sunlit road near-black,
    # so no single text colour is readable on both. A dimmed panel makes
    # the legend independent of whatever it happens to land on.
    panel_h = line_height * len(config.SURFACE_CLASSES) + 10
    panel = overlay[max(0, y - 18):y - 18 + panel_h, 0:x + 140]
    if panel.size:
        panel[:] = (panel * 0.35).astype(np.uint8)

    for surface in config.SURFACE_CLASSES:
        count = counts[surface.key]
        swatch_color = surface.color if count else (110, 110, 110)
        cv2.rectangle(overlay, (x, y - 11), (x + 16, y + 3), swatch_color, -1)
        text = f"{surface.key}: {count}"
        text_color = config.TEXT_COLOR if count else (150, 150, 150)
        # Dark outline first so the text stays readable over pale surfaces.
        for thickness, color in ((3, (0, 0, 0)), (1, text_color)):
            cv2.putText(overlay, text, (x + 24, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, thickness, cv2.LINE_AA)
        y += line_height
