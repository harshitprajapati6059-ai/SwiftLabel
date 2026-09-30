# Road Annotator — AI Pipeline (Phase 1)

## Purpose
Validate that an open-source, fully local AI model can automatically
generate an accurate **drivable-area** polygon from road photos, before
any web app, API, or database work is built on top of it.

This phase does NOT include a UI, backend, database, or deployment —
only the segmentation → polygon → preview pipeline.

## What gets annotated

These annotations are for **ADAS**, and the output is deliberately as
simple as it can be: **one class, one polygon per image.**

The polygon covers the whole annotated ground surface as a single
region — carriageway, bike path, walkway and green area all merged into
it. Objects standing on that ground (parked cars, poles, pedestrians,
street furniture) are **covered over, not cut around**: the polygon
describes the extent of the surface, not the silhouette of whatever
happens to be sitting on it in that frame.

### What is deliberately NOT annotated

Nothing painted **on** a surface, and nothing that is not a surface:

- lane markings, including broken/dashed white lines
- directional arrows (Zeichen 297 / 297.1) and any other direction cue
- crosswalk stripes
- cracks, seams, patches and other defects

A marking tells you about the paint, not about where the car may go, so
it is out of scope. The earlier `paint.py` / `arrows.py` / `lines.py` /
`crosswalk.py` modules that detected these have been removed rather than
switched off — if markings are ever needed they are a separate tool with
separate outputs, not a flag in this one.

**Everything annotated is on the ground plane.** That invariant is what
makes the output a map of where the vehicle may travel. It is also why
Cityscapes `vegetation` (tree and bush canopy) is not a source label:
canopy is overhead, and folding it in makes the polygon swallow the
whole sky on a tree-lined frame. Overhanging foliage is left unlabelled.

## The class table is the vocabulary

`config.SURFACE_CLASSES` is the only place the class is defined — its
label, id, overlay colour, drivability, and which of the *model's* class
names fold into it. It currently holds a single entry, `road`. Rename it
there if your CVAT task uses a different label; no other file names it,
and no code refers to a model class index.

That last part is why `MODEL_NAME` can be repointed at a checkpoint from
a different dataset without a code change: source labels are matched by
name, and unmatched names are ignored. The Mapillary Vistas names for
bike lane and pedestrian area are already listed, so a Vistas-trained
checkpoint picks them up automatically — Cityscapes has no bike-lane
class, which is why a cycle track currently only lands in the polygon if
its surface reads as road or sidewalk.

The table is still a *list*, and the rest of the pipeline is still
written for many classes. Adding entries back — to split walkway and
green area out again — is a config edit plus setting
`MERGE_INTO_SINGLE_POLYGON = False`, not a rewrite.

## Folder structure

```
config.py      all tunable parameters + the surface class table
segment.py     model loading + inference only (image -> surface label map)
polygon.py     mask cleaning, single-polygon merge, mask-to-polygon
preview.py     overlay rendering
utils.py       logging, IO, and the run_pipeline() orchestrator
test.py        single-image entry point
batch.py       folder entry point
export_cvat.py CVAT XML export (no model needed)

app.py         review UI: FastAPI server + static frontend
store.py       annotation state on disk — what is edited, what is approved
jobs.py        runs the pipeline off the request thread, with progress
static/        the UI itself (no build step; plain HTML/CSS/JS)

images/        input images
models/        cached Hugging Face weights (offline after first run)
output/
  masks/       <name>_mask.png   colour-coded classes, for eyeballing
               <name>_labels.png single-channel label map (class_id + 1)
  overlays/    visual verification images
  polygons/    annotation JSON + per-image metadata  (what the MODEL said)
  edited/      annotation JSON                       (what the HUMAN decided)
  thumbs/      gallery thumbnails
logs/          pipeline.log
```

## Usage

### Review UI (recommended)

```bash
python app.py                         # then open http://127.0.0.1:8000
```

Drop images onto the page, hit **Auto-annotate all**, then work through the
results correcting and approving each one, and export when done. The server
binds loopback only and has no authentication — it is not written to be
exposed to a network.

The editor: drag a vertex to move it, click an edge to insert one,
right-click a vertex to delete it, `N` to draw a new polygon, `A` to approve
and advance, `←`/`→` to move between images, `F` to fit. Edits autosave.

The **Tuning** panel re-runs a single image with a different merge kernel,
probability floor or simplification ratio. Those overrides last for the
session only — `config.py` is never written to, and the values actually used
are recorded in each image's `_metadata.json`.

### Command line

```bash
python test.py images/sample.jpg      # one image, verbose report
python batch.py ./images              # every supported image in a folder
python export_cvat.py                 # CVAT XML, every annotated image
python export_cvat.py --scope approved   # only images approved in the UI
```

## Approval, and why edits are a separate file

The pipeline writes to `output/polygons/` and never reads it back. The UI
writes to `output/edited/`, and that file is the human-owned copy: the
model's own output is never overwritten, so **Revert to AI** is just deleting
it.

Approval is stored in that file rather than inferred from it, because an
image the model got right is approved with *no edits at all*. "A human file
exists" and "a human has checked this" are different facts, and collapsing
them would make a perfect prediction indistinguishable from a frame nobody
has looked at.

Four states follow from that: `unannotated`, `auto` (model ran, unchecked),
`edited` (corrected, not yet signed off), `approved`.

**Re-running the model does not touch an annotation you have saved.** The new
prediction is written to `output/polygons/` as always, but the UI keeps
showing your version until you explicitly revert.

The first run downloads the SegFormer weights into `models/`. Every run
after that loads from that cache and never touches the network, so a
blocked or flaky connection cannot break it. Set `OFFLINE_ONLY = True`
in `config.py` to refuse the download step outright and fail with a
clear message instead.

For an image nobody has reviewed, `export_cvat.py` reads whatever
`batch.py` last wrote. **Re-run `batch.py` after changing the class table
or any tuning value**, otherwise the export reproduces the previous
configuration's polygons.

A reviewed image is exported straight from `output/edited/`, exactly as it
was drawn — re-deriving it from the cached mask would run the contour
approximation again and shift every vertex placed by hand. Tuning values
therefore have no effect on an image you have already corrected, which is
the intended behaviour: your edit is the annotation at that point.

## Output format

`output/polygons/<name>_polygon.json`:

```json
{
  "image_name": "sample.jpg",
  "image_width": 1600,
  "image_height": 1600,
  "classes": [
    { "class_id": 0, "label": "road", "drivable": true }
  ],
  "shapes": [
    { "id": 0, "class_id": 0, "label": "road", "drivable": true,
      "points": [ {"x": 1, "y": 643}, ... ],
      "bbox": [0, 795, 1600, 577], "area_px": 663214 }
  ]
}
```

`drivable` is carried on every shape so a consumer never has to
re-derive the ADAS bit by string-matching the label.

## How it works

1. **Segmentation** — SegFormer-B5 (Cityscapes) gives a softmax over its
   own 19 classes. Argmax is taken over **all** of them, not just the
   mapped ones — restricting it to a subset would force sky, buildings
   and cars into the surface class instead of letting them lose
   outright. Winners are then folded into our class via the config
   table; anything below `MIN_CLASS_PROBABILITY` is left **unlabelled**.

2. **Per-class cleaning** — opening, closing, bounded hole filling, small
   blob removal, then ego-vehicle exclusion.

3. **Overlap resolution** — contested pixels go to whichever class won
   the argmax there. A no-op with one class, and kept because it is what
   keeps the classes mutually exclusive the moment a second one is added
   back.

4. **Single-polygon merge** — union → close → fill all holes → keep the
   largest component → re-cut the ego vehicle.

5. **Polygon** — one simplified contour.

### Why the merge needs both a close and a hole fill

Two different kinds of gap have to disappear, and neither step reaches
the other's case:

- a car in the middle of the road leaves an **enclosed** hole, which
  only the hole fill reaches;
- a car parked against the kerb bites a notch that is **open to the
  outside**, which the hole fill cannot see and only the closing kernel
  can bridge.

Re-applying the ego mask afterwards is not optional. A 101 px closing
kernel will happily bridge straight across the vehicle hood, and filling
all holes would then seal it in — undoing the one exclusion every frame
depends on.

An enclosed hole is filled completely at any kernel size. An **edge**
notch is bridged progressively, so how much of a kerb-side car gets
covered scales with the kernel: measured on a 70 px bite, 51 px recovers
9%, 101 px recovers 50%, 161 px recovers 100%. The default of 101 is the
point where every disconnected island in the sample frames gets absorbed
while the boundary stays tight; raise it toward 161 if kerb-side cars
still leave visible bites and you would rather over-cover.

If a region survives all that still disconnected, it is dropped to keep
the output a single polygon, and the run **warns** with the pixel count
rather than dropping it silently.

### Vehicle hood exclusion

The camera sees the ego vehicle's hood/mount as a black region at the
bottom of every frame, and it must never end up inside the polygon.
`EGO_VEHICLE_MASK_POLYGON` is used only as a **search zone** (scaled to
the image, plus a margin); the actual near-black pixels are re-detected
per frame inside that zone. That adapts to the boundary drifting a few
pixels between frames without risking false positives on genuine shadows
elsewhere on the road.

It must run **last** — both in `clean_mask` and again after the merge.
Run before hole filling, the hood cutout gets misread as an enclosed
hole inside a region that spans the full image width, and filled
straight back in.

### Tuning

- `MERGE_CLOSE_KERNEL_PX` — the main knob. Raise it if the polygon still
  indents around parked cars or splits into pieces; lower it if the
  boundary starts bulging out over genuine background.
- `MERGE_INTO_SINGLE_POLYGON` — `False` returns one polygon per
  connected region.
- `MIN_CLASS_PROBABILITY` — raise to leave more of the frame unlabelled
  and only keep confident surface calls; lower to label more of it.
- `POLYGON_EPSILON_RATIO` — polygon simplification. Raise for fewer
  vertices, lower to hug the boundary more tightly.
- `MIN_CONTOUR_AREA_PX` / `MIN_BLOB_AREA_PX` — how small a patch still
  counts before the merge runs.
- `EGO_VEHICLE_BLACK_THRESHOLD` — raise if a lighter hood is not being
  fully excluded.

## Known limitations

- **Polygon holes are not exported.** The polygon is its outer boundary
  only; CVAT polygons have no native hole support. With
  `MERGE_FILL_ALL_HOLES = True` there are no holes to lose, which is
  part of why filling them is the default.
- **Disconnected ground is dropped**, not merged, so that "one polygon"
  stays true. Watch the warnings for the pixel count.
- **Overhanging foliage is unlabelled**, by design — see above.
- **Bike paths are not separately detectable** on Cityscapes weights.
  Moot while everything is one class; it matters again if the class
  table is split back up.
