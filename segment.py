"""
segment.py - Model loading and inference ONLY.
Responsibility: image in -> (surface label map, per-class confidence) out.
No polygon logic, no file saving, no drawing.

Isolated on purpose: to swap in a different segmentation model later
(a fine-tuned checkpoint, or a different architecture entirely), only
this file needs to change.

The model's own classes are never referenced by index. They are resolved
by NAME against config.SURFACE_CLASSES, so a checkpoint trained on a
different dataset works as long as its label names appear in that table.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

import config

# Value used in a label map for "no surface class here". Not 0, because 0
# is a real class id (road).
UNLABELLED: int = -1


def _load_offline_first(loader, model_name: str):
    """
    Load a pretrained component from the local cache, downloading only if
    it is genuinely not there yet.

    Without `local_files_only`, `from_pretrained` revalidates against the
    Hub on *every* run even when the weights are fully cached. That turns
    any network problem into a hard startup failure — a reset connection
    surfaces as "Can't load image processor ... make sure it is the
    correct path to a directory containing a preprocessor_config.json",
    which reads like a missing-file error rather than a network one.

    Trying the cache first makes the documented behaviour real (download
    once, then run fully offline) and removes a network round-trip from
    every subsequent startup.
    """
    try:
        return loader.from_pretrained(
            model_name, cache_dir=str(config.MODELS_CACHE_DIR), local_files_only=True
        )
    except OSError:
        # Not cached yet: this is the first run, so allow the download.
        if config.OFFLINE_ONLY:
            raise RuntimeError(
                f"'{model_name}' is not in the local cache at "
                f"{config.MODELS_CACHE_DIR} and OFFLINE_ONLY is set. Run once "
                f"with OFFLINE_ONLY = False (and a working connection) to "
                f"download the weights."
            ) from None
        return loader.from_pretrained(
            model_name, cache_dir=str(config.MODELS_CACHE_DIR)
        )


class SurfaceSegmenter:
    """
    Loads a semantic segmentation model once and maps its output onto the
    ADAS surface classes in config.SURFACE_CLASSES.
    """

    def __init__(self, model_name: str = config.MODEL_NAME) -> None:
        self.device = torch.device(
            "cuda" if (config.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        self.processor = _load_offline_first(SegformerImageProcessor, model_name)
        self.model = _load_offline_first(
            SegformerForSemanticSegmentation, model_name
        ).to(self.device)
        self.model.eval()

        id2label: dict[int, str] = self.model.config.id2label
        by_name = {str(label).lower().strip(): int(idx) for idx, label in id2label.items()}

        # Model class indices contributing to each of our surface classes.
        self.source_ids: dict[str, list[int]] = {}
        for surface in config.SURFACE_CLASSES:
            self.source_ids[surface.key] = [
                by_name[name.lower()] for name in surface.source_labels
                if name.lower() in by_name
            ]

        # A class the checkpoint simply cannot produce (bike_path on
        # Cityscapes) is not an error — it stays in the export vocabulary
        # for manual relabelling. But it must be visible, otherwise its
        # permanent absence reads as "no bike paths in this footage".
        self.unsupported_classes: list[str] = [
            surface.key for surface in config.SURFACE_CLASSES
            if not self.source_ids[surface.key]
        ]
        if len(self.unsupported_classes) == len(config.SURFACE_CLASSES):
            raise ValueError(
                f"None of the configured surface classes match this model's "
                f"labels. Model labels: {sorted(by_name)}"
            )

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """
        Run segmentation on a BGR OpenCV image.

        Returns:
            label_map: int16 array (H, W) holding the config class_id of
                       the surface at each pixel, or UNLABELLED (-1).
            confidences: mean softmax probability per surface class over
                         the pixels assigned to it. Classes with no pixels
                         are absent from the dict.
        """
        h, w = image_bgr.shape[:2]
        image_rgb = Image.fromarray(image_bgr[:, :, ::-1])

        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, num_classes, h', w')

        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)[0]  # (num_classes, H, W)

        # Argmax over ALL model classes, not just the mapped ones. Taking
        # it over a subset would force sky, buildings and cars into the
        # nearest surface class instead of letting them lose outright.
        top_probs, class_map = probs.max(dim=0)
        confident = top_probs >= config.MIN_CLASS_PROBABILITY

        label_map = np.full((h, w), UNLABELLED, dtype=np.int16)
        confidences: dict[str, float] = {}

        top_probs_np = top_probs.cpu().numpy()
        class_map_np = class_map.cpu().numpy()
        confident_np = confident.cpu().numpy()

        for surface in config.SURFACE_CLASSES:
            source_ids = self.source_ids[surface.key]
            if not source_ids:
                continue

            selected = np.zeros((h, w), dtype=bool)
            for source_id in source_ids:
                selected |= class_map_np == source_id
            selected &= confident_np

            if not selected.any():
                continue

            label_map[selected] = surface.class_id
            confidences[surface.key] = float(top_probs_np[selected].mean())

        return label_map, confidences
