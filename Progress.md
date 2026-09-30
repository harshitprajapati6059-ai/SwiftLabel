# 🗺️ Road Annotator Progress Dashboard

## 🎯 High-Level Objective
An offline-first AI pipeline that ingests road imagery and utilizes a Hugging Face `SegFormer-B5` model to generate highly precise binary segmentation masks, visual confidence overlays, and simplified vector polygons for streamlined freelance annotation work.

Annotation targets are the **road surface** (one polygon, from the model), every **painted directional arrow** on it, and every **broken white line** (both from classical CV). Arrow coverage spans two catalogue families: **Zeichen 297** (left, straight, right and the combined variants) and **Zeichen 297.1** (*Vorankündigungspfeil* advance warning arrows, left and right). A broken line is annotated as a **single polygon spanning the whole run**, gaps bridged — not one polygon per dash.

## 📊 System Overview & Tech Stack
- **Architecture:** SegFormer-B5 (Cityscapes) for dense pixel-level edge accuracy.
- **Marking Extraction:** Multi-scale morphological top-hat, run once per frame in `paint.py` and shared by both marking detectors. Not model-based — Cityscapes has no road-marking class to ask for.
- **Arrow Detection:** Convexity shape gate over the paint mask (`arrows.py`).
- **Broken Line Detection:** Permissive dash candidates + strict collinear chaining (`lines.py`).
- **Runtime Environment:** Transformers + PyTorch (Auto-fallback: CUDA GPU ➡️ CPU), OpenCV, Pillow.
- **Network Constraints:** 100% Offline execution after initial weight caching (`models/`).
- **Isolation Principle:** Pure inference is fully encapsulated inside `segment.py` to allow painless architectural or checkpoint swapping later.

---

## 🚀 Step-by-Step Inference Execution Guide
To run the road annotation pipeline on a targeted individual image, use the standard verification wrapper:

### Execution Syntax
```bash
python test.py images/testing.jpeg
```

### Underlying Processing Stream
1. **Environment Setup & Verification**
   - The CLI intercepts your path argument (`images/testing.jpeg`).
   - The script initializes configuration configurations from `config.py` and checks device availability.
2. **Model Ingestion & Caching (First-time Run)**
   - On its initial run, the pipeline hooks into the Hugging Face Hub to download the weights for `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`.
   - The weights are permanently routed and saved to the local `/models` cache directory. *Subsequent runs pull 100% offline from this folder.*
3. **Core Model Inference (`segment.py`)**
   - The targeted image is loaded as a BGR OpenCV array and converted to an RGB PIL Image.
   - The tensor passes through `SegformerImageProcessor` and is processed by the model on the available `DEVICE`.
   - The resulting logits are upscaled back to original dimensions using bilinear interpolation.
   - A binary road mask (0 or 255) is generated alongside a mean softmax probability confidence metric.
4. **Mask Post-Processing & Cleanup (`polygon.py`)**
   - The raw binary mask is passed to OpenCV for morphological cleaning (Closing to fill minor gaps, followed by Opening to remove speckle noise) based on `MORPH_KERNEL_SIZE`.
   - Connected component areas smaller than `MIN_BLOB_AREA_PX` are filtered out.
5. **Vector Conversion & Simplification (`polygon.py`)**
   - Contours are extracted from the clean mask.
   - If `KEEP_ONLY_LARGEST_CONTOUR` is active, the pipeline isolates the single largest road mass.
   - The Ramer-Douglas-Peucker (RDP) algorithm approximates and simplifies the contour curves into a vector polygon using `POLYGON_EPSILON_RATIO`.
6. **Arrow Detection & Vectorization (`arrows.py`)**
   - A multi-scale morphological top-hat on the CIELAB lightness channel isolates painted markings *inside the cleaned road mask*. Otsu's threshold is computed over road pixels only, so sky and foliage cannot skew it.
   - Interiors are refilled per connected component, so markings wider than the largest kernel are not left hollow.
   - Each candidate is passed through a shape gate (area, extent, solidity, convexity-defect depth, aspect ratio, stroke thickness) that separates arrows from lane lines, dashes and stop bars.
   - Every accepted arrow is re-cut from an upscaled crop of itself, then RDP-simplified with `ARROW_POLYGON_EPSILON_RATIO`.
7. **Visualization & Visual Rendering (`preview.py`)**
   - The raw image background receives an alpha-blended color mask overlay (`OVERLAY_MASK_COLOR`).
   - The calculated simplified vector polygon outline, vertices, and pipeline confidence metrics text are drawn onto a preview canvas.
   - Each arrow is outlined in its own color from `ARROW_INSTANCE_COLORS` and labelled `arrow_1`, `arrow_2`, ... so neighbouring polygons stay distinguishable during review.
8. **Disk Serialization & Output Generation**
   - The results are written out to the `/output` directories simultaneously:
     - `/output/masks/testing_mask.png`: Raw 1-channel binary png mask.
     - `/output/overlays/testing_overlay.png`: Visual alpha-blended verification image.
     - `/output/polygons/testing_polygon.json`: All polygons (road + arrows), each with its own `class_id`.
     - `/output/polygons/testing_metadata.json`: Run telemetry and per-arrow shape statistics.
   - Execution telemetry is appended directly to `/logs/pipeline.log`.

---

## 🗺️ Implementation Roadmap

### 🟢 Phase 1: Architecture, Configuration & Core Inference (COMPLETE)
- [x] Establish isolated directory structure (`models/`, `images/`, `output/`, `logs/`)
- [x] Create centralized `config.py` tracking parameters (Morphology, Contours, Thresholds, BGR Colors)
- [x] Implement runtime-fallback device logic (GPU `cuda` context automatically dropping to `cpu`)
- [x] Finalize `segment.py` with `RoadSegmenter` class tracking:
  - Hugging Face offline cache routing to `models/`
  - Dynamic ID scanning for the native `"road"` class label
  - Interpolated bilinear tensor sizing matching target dimensions
  - Dynamic road-surface pixel mean softmax confidence computation

### 🟢 Phase 2: Vectorization & Image Processing (COMPLETE)
- [x] Complete `polygon.py` data stream extraction
  - [x] Implement mathematical morphology operations (Closing/Opening iterations) via `config.py`
  - [x] Write contours reduction loop leveraging `POLYGON_EPSILON_RATIO`
  - [x] Apply `KEEP_ONLY_LARGEST_CONTOUR` optimization filter and target minimum area limits
- [x] Complete `preview.py` to map semantic outputs back into BGR canvas alpha-blended visualizations
- [x] Complete `utils.py` for standard disk I/O and runtime exception telemetry logging

### 🟢 Phase 2b: Road Arrow Annotation (COMPLETE — 2026-07-24)
- [x] Build `arrows.py`: paint extraction, shape gate, ROI-upscaled polygon refinement
- [x] Add the full arrow parameter block to `config.py`
- [x] Render one distinctly-coloured, labelled polygon per arrow in `preview.py`
- [x] Export multi-shape annotation JSON with a `class_id` per polygon, keeping the legacy `polygon` key
- [x] Validate on the real test frame: 2/2 arrows found, 0 false positives across ~14 competing paint blobs
- [x] Validate synthetically on all six Zeichen 297 variants at 40/80/160 px: **6/6 recall at every scale, 0 false positives** against lane-line, dashed-line and stop-bar distractors

### 🟢 Phase 2c: Zeichen 297.1 Advance Warning Arrows (COMPLETE — 2026-07-24)
- [x] Profile the 297.1 geometry against the existing gate; found `thickness_ratio` and `aspect_ratio` both break down as the shaft elongates
- [x] Replace `thickness_ratio` with `defect_thickness_ratio` (notch depth ÷ widest point) as the primary discriminator — elongation-invariant by construction
- [x] Raise `ARROW_MAX_ASPECT_RATIO` 8 → 16; demote `ARROW_MIN_DEFECT_RATIO` to a redundant second opinion at 0.15
- [x] Validate across 3 lengths × 5 elongations (shaft 6x to 34x): **15/15 pass**
- [x] Validate left + right, solid and outlined head, 3 scales: **12/12 pass**; polygons trace shaft and angled head with flanking lane lines correctly ignored
- [x] Confirm no regression: 297 family still 6/6 at every scale with 0 false positives; real test frame still 2/2 arrows, 0 false positives

### 🟢 Phase 2d: Broken (Dashed) White Lines (COMPLETE — 2026-07-24)
- [x] Extract shared primitives into `paint.py`; extraction now runs once per frame and both detectors consume the same mask
- [x] Build `lines.py`: permissive dash candidates, collinear chaining, one band polygon per line
- [x] Invert the usual precision strategy — loose per-dash gate, strict grouping — after finding a tight gate discarded the entire far half of every line
- [x] Exclude arrow-claimed pixels from the dash pool rather than re-deriving an arrow test at dash scale
- [x] Wire through preview (own palette, red excluded), JSON export (`line_N`, class ids from 101), metadata and both CLIs
- [x] Validate synthetically: parallel lines stay separate, solid lines rejected, curved line followed, perspective-scaled dashes chained — **4/4 scenarios**
- [x] Real test frame: **2 broken lines** (13 dashes + 3 dashes), matching the reference annotation; arrows and road unchanged
- [x] **Reworked the band into a clean centreline ribbon (2026-07-25)** after user feedback that the traced-outline band was too lumpy; sharp near-parallel edges following the arc, all 4 synthetic scenarios still pass

### 🔴 Phase 3: Batch Optimization & Verification (PENDING)
- [x] Code `batch.py` to handle directory scanning against `SUPPORTED_EXTENSIONS` without blocking loops
- [x] Build functional entry point validations in `test.py` to intercept single image terminal arguments
- [/] Finalize local dependencies profiling (`requirements.txt`) and operational validation steps (`README.md`)
- [ ] Validate arrows across a wider image set — near-field arrows, wet/night frames, and worn paint are all still unexercised on real data

---

## ⚠️ Architectural Flags & Active Issues

### 🎛️ Config Arbitrations (Flagged for Review)
- **Multi-Region Splitting:** `KEEP_ONLY_LARGEST_CONTOUR = True` is forced globally by default. If a foreground element (e.g., a traffic sign, vehicle, or bridge pillar) cuts across a single continuous road segment, the smaller split section will be discarded and generate a pipeline warning log. 
- **Hardware Agnostics:** `DEVICE = "cuda"` auto-falls back to `"cpu"`. Heavy `b5` computational scaling means CPU passes will experience bottleneck latencies during long batch pipelines.
- **Merged-Marking Splitting:** `ARROW_SPLIT_MERGED = False` by default. It recovers an arrow fused to a lane line, but under a wide FOV the lane lines converge into one paint web in nearly every frame, and eroding that web yields elbow-shaped fragments that score like arrowheads — two false positives per frame on the test image. Enable only if arrows are demonstrably being lost to fusion.
- **Marking Recall Is Gated By The Road Mask:** arrows and lines are searched for only inside the cleaned road mask, so any road under-segmentation silently caps marking recall.
- **Broken Lines Fade Out Toward The Horizon:** dashes below `LINE_DASH_MIN_AREA_PX` (12 px) are dropped, so a line's band stops short of the vanishing point rather than tapering to nothing. Lowering the floor extends reach at the cost of admitting speckle.
- **Solid Markings Are Not Annotated:** solid lane lines, stop bars and the thick painted arcs are deliberately rejected (a solid line is a chain of one). If they ever need annotating, that is a new class, not a loosened threshold.

### 🛑 Active Blockers
- *None.*

---

## 🪵 Engineering Decision Log

*   **2026-07-25: Pedestrian Crossings Suppressed As A Region, Not Rejected Per Blob**
    Zebra crossings were the dominant false-arrow source: their bars, and the worn fragments around them, carry shallow notches under the fisheye that clear the arrowhead test by a hair, and their shape numbers are a dead heat with a real arrow (real straight arrow `solidity 0.70 / defect_thickness 0.36 / defect_ratio 0.29` vs a zebra fragment `0.75 / 0.36 / 0.28`). No per-shape threshold can separate them. The separable signal is *arrangement*: a crossing is a cluster of several LARGE, ELONGATED bars packed side by side, which neither a lone arrow nor a much-smaller lane dash produces. New `crosswalk.py` clusters such bars (area ≥ 700, length ≥ 90, aspect ≥ 3; single-link by centre distance ≤ 1.2× bar length; ≥ 3 bars) and hands back their footprint — the bar pixels, their convex hull (to cover the fragments stranded in the gap between the two fanned arms the fisheye splits a crossing into), dilated. `utils.run_pipeline` subtracts that region from the paint mask once, before both the arrow and line gates, so the whole crossing never reaches them. Result on the six test frames: the two zebra scenes went from 4 and 2 false arrows to **0/0**, with the genuine arrows on the other frames untouched.
*   **2026-07-25: Arrow Gate Tightened On Three Axes That Cost No Real Arrows**
    Alongside crossing suppression, three gates were tightened, each validated against the confirmed real arrows so recall was preserved. `ARROW_MIN_EXTENT` 0.25→0.30 (a converging lane-line web at a curb sat at 0.28; real arrows are 0.30–0.45). `ARROW_MAX_SOLIDITY` 0.88→0.80 (nothing real lived in 0.80–0.88 — dashes are 0.84+ and the largest real arrow measured 0.75 — only chunky convex patches and merged bars). `ARROW_MIN_DEFECT_COUNT` 1→2 (an arrowhead has two barbs flanking the shaft, so every confirmed arrow shows ≥ 2 significant notches; a single-notch blob is far more often an incidental sliver on a car edge or pavement seam). Combined effect across the six frames: false arrows fell from ~14 to ~4, the residual being isolated road-boundary slivers and one wheelchair symbol — not the systematic crossing failure the user reported.

*   **2026-07-23: SegFormer-B5 Over DeepLabV3+**  
    Selected SegFormer-B5 despite its larger parameter weight to achieve maximum road edge boundary definition. DeepLabV3+ fallback retained purely as an architectural alternative for ultra-low computational specifications.
*   **2026-07-23: SegFormer-B5 Over Mask2Former**  
    Dismissed Mask2Former despite slightly higher benchmark marks. The complex query-based mask decoder structure adds substantial overhead and presents significantly harder paths toward downstream structural model adjustments/fine-tuning.
*   **2026-07-24: Classical CV Over A Second Model For Arrows**  
    Cityscapes has no road-marking class, so arrows cannot come from SegFormer. The alternatives were an open-vocabulary detector (OWLv2/GroundingDINO) plus SAM, or classical CV. Chose classical CV: road paint is an almost ideal classical target — a bright, thin, high-contrast structure on a locally uniform dark surface whose location is already narrowed by the road mask. It also stays deterministic, tunable, adds no weights to the offline cache, and costs milliseconds rather than a second model pass. The open-vocab route remains the fallback if real-world recall proves insufficient.
*   **2026-07-24: Top-Hat Over Unsharp-Mask For Paint Extraction**  
    A local-mean difference was tried first and rejected. It fires on asphalt aggregate texture, so the threshold needed to suppress that noise was already high enough to erase faint far-field arrows entirely. Morphological top-hat encodes the actual prior — *brighter than its surroundings AND narrower than the kernel* — and cleanly separated both.
*   **2026-07-24: Convexity Defects As The Arrow Discriminator**  
    Arrows are told from lane lines, dashes and stop bars by concavity, not by size or brightness. A dash is a convex blob; an arrow always has deep notches where the barbs meet the shaft. Measured on the test frame, arrow defect depth normalised by √area is 0.29–0.51 while every dash is below 0.10 — a wide, stable gap to place a threshold in.
*   **2026-07-24: Notch Depth Normalised By Thickness, Not √Area (297.1 support)**  
    Adding the elongated `Vorankündigungspfeil` broke two features at once: `thickness_ratio` (head width ÷ length) and `aspect_ratio` both degrade monotonically as the shaft lengthens, and a realistically-proportioned advance arrow fails both. Measured across 3 lengths × 5 elongations, `solidity` (0.51–0.55) and `extent` (0.32–0.35) held completely steady, and notch depth ÷ *widest point* stayed in 0.58–1.53 versus 0.10–0.23 for dashes. Switched the primary gate to that ratio: it asks whether the notch is deep relative to the head it is cut into, which is a fixed property of arrow geometry at any length. `thickness_ratio` was dropped from the gate but is still reported for diagnostics.
*   **2026-07-24: Taper Ratio Evaluated And Rejected**  
    `max_thickness × length ÷ area` looked like the natural "does it have a localised head" feature, and it cleanly killed the merged-web false positives (1.17–1.23) that `thickness_ratio` let through. Measuring it on real 297.1 geometry killed the idea: legitimate advance arrows score 1.16–1.52, overlapping the false positives exactly. Recorded here so it is not re-proposed.
*   **2026-07-24: Cache-First Weight Loading (offline claim made real)**  
    `from_pretrained` was called without `local_files_only`, so it revalidated against the Hub on *every* run despite the weights being fully cached — the "100% offline after initial caching" claim in this document was aspirational, not enforced. A reset connection then surfaced as `Can't load image processor ... make sure it is the correct path to a directory containing a preprocessor_config.json`, which reads like a missing-file error and sends you looking in entirely the wrong place. Now loads from cache first and only downloads when the weights are genuinely absent, which also removes a network round-trip from every startup.
*   **2026-07-24: Broken Lines — Loose Gate, Strict Grouping**  
    The first attempt filtered dashes individually and lost the entire far half of every line: a horizon dash is a ~20 px sliver whose solidity and extent are dominated by pixel quantisation, and a per-dash gate tight enough to be safe throws most of the line away. Inverted it — admit candidates cheaply, then demand a chain of ≥3 collinear neighbours. Noise does not produce a run of evenly-spaced, consistently-oriented blobs along a common path, so the arrangement is a far stronger signal than any single blob's shape. As a free consequence, a solid lane line is rejected for being a chain of one, with no extra test needed.
*   **2026-07-24: Exclude Arrow Pixels Rather Than Re-Test For Arrows**  
    `lines.py` initially re-used the arrow notch test to avoid claiming arrows as dashes. That silently deleted the far end of every line: on a sliver 1-2 px wide, `max_thickness` ≈ 1, so any single-pixel wobble in the contour scores as a deep arrowhead notch. Arrow detection now runs first and hands its shapes over as an exclusion mask, which is both correct at every scale and removes a duplicated test.
*   **2026-07-25: Centreline Ribbon, Not A Traced Paint Outline**  
    The line polygon is now a constant-following-width ribbon: a centreline fitted through the dash centres (nearest-neighbour ordering to follow the curve, light Chaikin smoothing, ends extended half a dash) offset by the local half-width to give two near-parallel edges. This replaced tracing the outline of the dashes-plus-bridges, which the user rejected as "really inaccurate" — that outline followed the paint's own ragged silhouette, so it bulged around every dash and pinched at every gap. Their reference sketch was explicit: a sharp rectangular corridor with the dashes inside, not a wavy organic blob. Half-width tapers with the dashes so the ribbon stays tight under perspective. (Supersedes the earlier pairwise-hull note; the hull approach was itself a fix for one-hull-cuts-the-corner, but both traced paint and both were lumpy.)
*   **2026-07-24: Per-Component Hole Filling, Not Border Flood Fill**  
    Refilling hollow large arrows with `polygon.py`'s border-flood-fill trick flooded the entire lower image solid: a lane line spanning the full frame width cuts the image in two, so everything beyond it counts as "unreachable from the border". Re-drawing each external contour filled has no such global dependency. Same class of bug as the ego-vehicle/`FILL_HOLES` ordering hazard already documented in `clean_mask`.
*   **2026-07-23: Native Cityscapes Classes Over YOLOP/TwinLiteNet**  
    Rejected specialized driving models like YOLOP/TwinLiteNet. These models output highly restrictive specific "drivable area" boundaries rather than absolute material road surfaces. They additionally pack unneeded concurrent tracking heads (e.g., lane indicators or traffic actors) that introduce overhead and match poorly against versatile, multi-angle freelance annotation assets.
