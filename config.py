"""
config.py - Central configuration for the Road Annotator AI pipeline.
Nothing in the rest of the codebase should hardcode values that live here.

SCOPE (ADAS surface annotation)
-------------------------------
This pipeline annotates DRIVABLE SURFACE TYPE only. The question every
polygon answers is "can a car go here?" — road yes, walkway/bike path/
green area no.

It deliberately does NOT annotate anything painted ON a surface:
no lane markings, no dashed/broken lines, no directional arrows, no
crosswalk stripes, no cracks. Those are properties of the paint, not of
where the vehicle may drive, and they were removed from this pipeline on
purpose. Do not add them back here — a new marking pipeline would be a
separate tool with its own outputs.
"""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

IMAGES_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "output"
MASKS_DIR = OUTPUT_DIR / "masks"
OVERLAYS_DIR = OUTPUT_DIR / "overlays"
POLYGONS_DIR = OUTPUT_DIR / "polygons"
LOGS_DIR = BASE_DIR / "logs"
MODELS_CACHE_DIR = BASE_DIR / "models"

# The human-owned copy of an annotation, written by the review UI. Kept
# apart from POLYGONS_DIR so the model's own output is never overwritten
# and "revert to AI" is just deleting a file. Its presence is also what
# marks an image as reviewed — see store.py.
EDITED_DIR = OUTPUT_DIR / "edited"

# Downscaled JPEGs for the gallery grid. The source frames are hundreds of
# KB each, so a grid of a hundred would be tens of MB per page load.
THUMBS_DIR = OUTPUT_DIR / "thumbs"

for _d in (IMAGES_DIR, OUTPUT_DIR, MASKS_DIR, OVERLAYS_DIR, POLYGONS_DIR,
           EDITED_DIR, THUMBS_DIR, LOGS_DIR, MODELS_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# SegFormer fine-tuned on Cityscapes. Swap to
# "nvidia/segformer-b0-finetuned-cityscapes-1024-1024" for a much
# faster/lighter model at the cost of boundary precision, or keep b5 for
# best accuracy. Only this value needs to change to try another
# checkpoint — the class table below is matched against the checkpoint's
# own id2label names, so nothing in the code refers to a class index.
MODEL_NAME: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
DEVICE: str = "cuda"  # falls back to "cpu" automatically if no GPU is found

# Weights are always loaded from MODELS_CACHE_DIR first and only fetched
# from the Hub when they are genuinely absent, so a network problem can
# never break an already-cached run. Set True to refuse the download step
# outright and fail with a clear message instead — useful on a machine
# that is meant to stay offline, where a silent download would be a bug.
OFFLINE_ONLY: bool = False

# ---------------------------------------------------------------------------
# Surface classes — the entire annotation vocabulary
# ---------------------------------------------------------------------------
# This table IS the label set. Adding, renaming or recolouring a class is
# a one-line edit here; no other file names a class.
#
# `source_labels` are matched case-insensitively against the checkpoint's
# id2label values, and several source classes may fold into one of ours
# (grass and trees are both just "not drivable, green"). Names from more
# than one dataset are listed on purpose so that pointing MODEL_NAME at a
# different checkpoint keeps working without a code change — unmatched
# names are simply ignored.
#
# `drivable` is the ADAS bit the whole exercise exists for and is carried
# through to the exported JSON, so a consumer never has to re-derive
# "can the car go here" from a label string.


@dataclass(frozen=True)
class SurfaceClass:
    key: str                          # exported label
    class_id: int                     # exported class_id
    source_labels: tuple[str, ...]    # model label names folded into this class
    color: tuple[int, int, int]       # overlay colour, BGR (OpenCV convention)
    drivable: bool


# ONE class. Road, bike path, walkway and green area are all folded into
# it, and MERGE_INTO_SINGLE_POLYGON below then reduces the whole frame to
# a single polygon covering the lot.
#
# Rename the class in one place — `key` below — if your CVAT task uses a
# different label. Nothing else in the codebase names it.
#
# The source labels are the GROUND PLANE only. Cityscapes "vegetation"
# (tree and bush canopy) is deliberately not among them: it is vertical
# and mostly above the horizon, so on a tree-lined frame the canopy
# prediction covers the bare branches, the gaps of sky between them close
# up under morphology, and the polygon ends up swallowing the entire sky.
# Objects standing ON the ground (cars, poles, pedestrians) do not need
# to be listed — the merge step fills straight over them.
SURFACE_CLASSES: tuple[SurfaceClass, ...] = (
    SurfaceClass(
        key="road",
        class_id=0,
        source_labels=(
            # carriageway
            "road",
            # walkway / pedestrian area
            "sidewalk",
            "pedestrian area",
            "construction--flat--sidewalk",
            "construction--flat--pedestrian-area",
            # bike path (absent from Cityscapes; present in Mapillary Vistas)
            "bike lane",
            "bike-lane",
            "construction--flat--bike-lane",
            # green area at ground level — grass, soil, roadside banks
            "terrain",
            "nature--terrain",
        ),
        color=(0, 200, 255),          # amber
        drivable=True,
    ),
)

SURFACE_CLASSES_BY_KEY: dict[str, SurfaceClass] = {c.key: c for c in SURFACE_CLASSES}

# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------
# A pixel is labelled with the winning class only if that class's softmax
# probability clears this. Everything below is left UNLABELLED rather than
# forced into the nearest surface — an unlabelled pixel is an honest "the
# model does not know", which is reviewable; a confidently wrong drivable
# label is not.
#
# There is deliberately no top-2 margin gate here (the old road-only
# pipeline had one to stop the road bleeding over the curb). Now that the
# surface on the other side of the curb is a class in its own right, the
# argmax boundary between them IS the annotation we want — a margin would
# only carve an unlabelled no-man's-land along every kerb line.
MIN_CLASS_PROBABILITY: float = 0.50

# Mean-probability floor below which a class's regions are flagged in the
# run warnings. Does not drop anything — it only asks for a human look.
CONFIDENCE_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Mask cleaning (applied per class)
# ---------------------------------------------------------------------------
MORPH_KERNEL_SIZE: int = 7
MORPH_CLOSE_ITERATIONS: int = 2
MORPH_OPEN_ITERATIONS: int = 1
MIN_BLOB_AREA_PX: int = 600          # blobs smaller than this are dropped

# Holes SMALLER than this are filled; larger ones are kept as genuine
# cutouts. A blanket fill was safe when road was the only class, but a
# green area legitimately wraps around a flower bed or a path, and
# filling that swallows the thing it surrounds. Set to 0 to fill nothing.
FILL_HOLES_MAX_AREA_PX: int = 4000

# ---------------------------------------------------------------------------
# Polygon extraction
# ---------------------------------------------------------------------------
POLYGON_EPSILON_RATIO: float = 0.002   # fraction of contour perimeter
MIN_CONTOUR_AREA_PX: int = 800
MAX_REGIONS_PER_CLASS: int = 0         # 0 = unlimited, else keep N largest

# ---------------------------------------------------------------------------
# Single-polygon merge
# ---------------------------------------------------------------------------
# Collapse the whole frame to ONE polygon covering all annotated ground,
# rather than one polygon per disjoint patch. Objects standing on that
# ground — parked cars, poles, pedestrians, street furniture — are
# covered over rather than cut around, which is the point: the polygon
# describes the extent of the surface, not the silhouette of what happens
# to be sitting on it that frame.
#
# Set False to go back to one polygon per connected region.
MERGE_INTO_SINGLE_POLYGON: bool = True

# Closing kernel used to bridge gaps before merging. This is what joins
# regions a foreground object has cut in two — a car parked half on the
# kerb bites a notch out of the boundary that hole filling cannot reach,
# because the notch is open to the outside rather than enclosed.
#
# Roughly the widest gap it can bridge, so it has to exceed the apparent
# width of a near-field car. Measured on the sample frames: at 51 two
# regions per set stayed islanded and were dropped; 101 absorbed every
# one while costing only 0.3pp of extra coverage. Past 101 coverage keeps
# creeping up with nothing left to gain — that is the boundary bulging
# out over background, so this is about as large as is useful.
#
# Raise it if the polygon still indents around parked cars or sheds
# islands (the run warns when it does); lower it if the boundary starts
# bulging away from the real edge of the surface.
MERGE_CLOSE_KERNEL_PX: int = 101

# Fill every enclosed hole regardless of size, so a car surrounded by
# road does not leave a doughnut. This deliberately overrides
# FILL_HOLES_MAX_AREA_PX, whose bounded behaviour exists to protect
# genuine cutouts that only matter when regions stay separate.
MERGE_FILL_ALL_HOLES: bool = True

# ---------------------------------------------------------------------------
# Preview / overlay rendering
# ---------------------------------------------------------------------------
# Each class is drawn in its own colour (see SURFACE_CLASSES above): a
# translucent fill so the extent is visible at a glance, plus a solid
# outline so the polygon boundary being reviewed stays crisp.
OVERLAY_FILL_ALPHA: float = 0.35
OVERLAY_OUTLINE_THICKNESS: int = 2
OVERLAY_DRAW_VERTICES: bool = False
OVERLAY_VERTEX_RADIUS: int = 3
OVERLAY_DRAW_LEGEND: bool = True
TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LOG_FILE: Path = LOGS_DIR / "pipeline.log"

# ---------------------------------------------------------------------------
# Ego-vehicle exclusion mask
# ---------------------------------------------------------------------------
# The vehicle hood/mount is always roughly in this region, but its exact
# pixel boundary shifts slightly frame to frame. So we use the polygon
# below only as a SEARCH ZONE (with a safety margin) and detect the real
# black pixels fresh, per image, inside that zone only. This keeps
# precision (adapts to per-frame variance) without risking false
# positives on real shadows elsewhere on the road.
#
# Applied to EVERY surface class, not just road: the hood is not a
# surface of any kind and must never end up inside an exported polygon.
EGO_VEHICLE_MASK_ENABLED: bool = True
EGO_VEHICLE_MASK_REFERENCE_SIZE: tuple[int, int] = (765, 763)  # (width, height) it was traced on
EGO_VEHICLE_MASK_POLYGON: list[tuple[int, int]] = [
    (1, 643), (1, 762), (764, 762), (764, 645),
    (594, 613), (452, 512), (310, 518), (153, 626),
]
EGO_VEHICLE_SEARCH_MARGIN_PX: int = 25   # buffer added around the polygon to cover pixel drift
EGO_VEHICLE_BLACK_THRESHOLD: int = 20    # grayscale value below which a pixel counts as "hood black"
