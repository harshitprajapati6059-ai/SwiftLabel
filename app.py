"""
app.py - Local annotation server. Serves the review UI and its API.

    python app.py            then open http://127.0.0.1:8000

Binds the loopback interface only. Nothing here is written to be exposed to
a network: there is no authentication, and every route can read and write
files under the project directory. If this ever needs to be reachable from
another machine, it needs an auth layer first.

Layout:
    store.py   what is on disk and what state each image is in
    jobs.py    running the pipeline off the request thread
    app.py     HTTP surface over both, plus the static UI
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import export_cvat
import jobs
import store
from utils import logger

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Queued rather than awaited: the port opens immediately so the page
    # loads and can show "warming up", instead of the browser failing to
    # connect for the ~14s the weights take to load.
    jobs.warm_up()
    logger.info(f"Annotation UI serving {config.IMAGES_DIR}")
    yield
    jobs.shutdown()


app = FastAPI(title="Road Annotator", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------

@app.exception_handler(store.UnsafeImageName)
async def _unsafe_name(request: Request, exc: store.UnsafeImageName) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": f"Invalid image name: {exc}"})


@app.exception_handler(store.ImageNotFound)
async def _not_found(request: Request, exc: store.ImageNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f"No such image: {exc}"})


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class Point(BaseModel):
    x: float
    y: float


class Shape(BaseModel):
    label: str
    points: list[Point]


class AnnotationBody(BaseModel):
    shapes: list[Shape]
    width: Optional[int] = None
    height: Optional[int] = None


class JobBody(BaseModel):
    # Empty means "every image that has no annotation yet" — the one-button
    # case. An explicit list is how re-running a single image works.
    images: Optional[list[str]] = None
    overrides: dict[str, Any] = Field(default_factory=dict)


class ExportBody(BaseModel):
    scope: str = "approved"


# ---------------------------------------------------------------------------
# Status and vocabulary
# ---------------------------------------------------------------------------

@app.get("/api/status")
def read_status() -> dict[str, Any]:
    return {
        "model": jobs.model_status(),
        "counts": store.counts_by_state(),
        "active_job": jobs.active_job(),
        "images_dir": str(config.IMAGES_DIR),
    }


@app.get("/api/classes")
def read_classes() -> list[dict[str, Any]]:
    """
    The annotation vocabulary, straight from config.SURFACE_CLASSES.

    The UI builds its class picker from this, so splitting the class table
    back into road/walkway/green_area needs no frontend change at all.
    Colours are stored BGR for OpenCV and reversed to RGB hex here.
    """
    out = []
    for surface in config.SURFACE_CLASSES:
        blue, green, red = surface.color
        out.append({
            "key": surface.key,
            "class_id": surface.class_id,
            "drivable": surface.drivable,
            "color": f"#{red:02x}{green:02x}{blue:02x}",
        })
    return out


@app.get("/api/settings")
def read_settings() -> dict[str, Any]:
    """Current values of the tunable knobs, so the panel can show defaults."""
    return {
        "tunable": {
            key: getattr(config, key) for key in jobs.TUNABLE
        },
        "overlay_alpha": config.OVERLAY_FILL_ALPHA,
    }


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

@app.get("/api/images")
def read_images() -> list[dict[str, Any]]:
    return store.summarise_all()


@app.get("/api/images/{name}/file")
def read_image_file(name: str) -> FileResponse:
    return FileResponse(store.resolve_image(name))


@app.get("/api/images/{name}/thumb")
def read_thumbnail(name: str) -> FileResponse:
    thumb = store.thumbnail(store.resolve_image(name))
    if thumb is None:
        raise HTTPException(status_code=422, detail="Could not decode image")
    return FileResponse(thumb, media_type="image/jpeg")


@app.get("/api/images/{name}/annotation")
def read_annotation(name: str) -> dict[str, Any]:
    image_path = store.resolve_image(name)
    document, origin = store.read_annotation(image_path)
    return {
        "name": image_path.name,
        "annotation": document,
        "origin": origin,
        "state": store.state(image_path),
    }


@app.put("/api/images/{name}/annotation")
def write_annotation(name: str, body: AnnotationBody) -> dict[str, Any]:
    image_path = store.resolve_image(name)
    shapes = [
        {"label": s.label, "points": [{"x": p.x, "y": p.y} for p in s.points]}
        for s in body.shapes
    ]
    try:
        document = store.save_annotation(image_path, shapes, body.width, body.height)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return {"annotation": document, "state": store.state(image_path)}


@app.delete("/api/images/{name}/annotation")
def revert_annotation(name: str) -> dict[str, Any]:
    image_path = store.resolve_image(name)
    reverted = store.revert(image_path)
    document, origin = store.read_annotation(image_path)
    return {
        "reverted": reverted,
        "annotation": document,
        "origin": origin,
        "state": store.state(image_path),
    }


@app.post("/api/images/{name}/approve")
def approve_image(name: str) -> dict[str, Any]:
    image_path = store.resolve_image(name)
    try:
        document = store.approve(image_path)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return {"annotation": document, "state": store.state(image_path)}


@app.delete("/api/images/{name}/approve")
def unapprove_image(name: str) -> dict[str, Any]:
    image_path = store.resolve_image(name)
    document = store.unapprove(image_path)
    if document is None:
        raise HTTPException(status_code=409, detail="Image is not approved")
    return {"annotation": document, "state": store.state(image_path)}


@app.delete("/api/images/{name}")
def remove_image(name: str) -> dict[str, Any]:
    image_path = store.resolve_image(name)
    store.delete_image(image_path)
    return {"deleted": name}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_images(files: list[UploadFile]) -> dict[str, Any]:
    saved: list[str] = []
    rejected: list[dict[str, str]] = []

    for upload in files:
        # Take only the basename: a browser sending a webkitRelativePath, or
        # a crafted request, must not be able to choose a directory.
        original = Path(upload.filename or "").name
        if not original:
            rejected.append({"name": str(upload.filename), "reason": "no filename"})
            continue
        if Path(original).suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            rejected.append({
                "name": original,
                "reason": f"unsupported type (allowed: "
                          f"{', '.join(config.SUPPORTED_EXTENSIONS)})",
            })
            continue

        target = _unique_path(config.IMAGES_DIR / original)
        try:
            target.write_bytes(await upload.read())
        except OSError as exc:
            rejected.append({"name": original, "reason": str(exc)})
            continue
        saved.append(target.name)

    return {"saved": saved, "rejected": rejected}


def _unique_path(path: Path) -> Path:
    """
    Never overwrite an existing image on upload.

    Re-dropping a folder is a normal accident, and silently replacing a file
    would also orphan the annotation already keyed to that name.
    """
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem} ({index}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise HTTPException(status_code=409, detail=f"Too many copies of {path.name}")


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@app.post("/api/jobs")
def create_job(body: JobBody) -> dict[str, Any]:
    if body.images is None:
        names = [
            p.name for p in store.list_images()
            if store.state(p) == store.UNANNOTATED
        ]
    else:
        names = [store.resolve_image(n).name for n in body.images]

    if not names:
        raise HTTPException(status_code=409, detail="Nothing to annotate")

    try:
        job_id = jobs.submit(names, body.overrides)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    return {"job_id": job_id, "images": names}


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    return {"cancelled": jobs.cancel(job_id)}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@app.get("/api/export/preview")
def export_preview() -> dict[str, Any]:
    """Counts for the export dialog, so it can show what each scope includes."""
    counts = store.counts_by_state()
    return {
        "counts": counts,
        "approved": counts[store.APPROVED],
        "annotated": counts[store.APPROVED] + counts[store.EDITED] + counts[store.AUTO],
        "unannotated": counts[store.UNANNOTATED],
    }


@app.post("/api/export/cvat")
def export_cvat_xml(body: ExportBody) -> Response:
    try:
        root, summary = export_cvat.build_export(body.scope)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    if root is None:
        raise HTTPException(
            status_code=409,
            detail=f"No images match scope '{body.scope}'. "
                   f"{'Approve some images first.' if body.scope == 'approved' else ''}",
        )

    buffer = io.BytesIO()
    ET.ElementTree(root).write(buffer, encoding="utf-8", xml_declaration=True)

    # Also written to output/ so the CLI and the UI leave the file in the
    # same place, and a download the browser discards is still recoverable.
    export_cvat.OUTPUT_XML.write_bytes(buffer.getvalue())
    logger.info(
        f"Exported {summary['image_count']} images (scope={body.scope}, "
        f"{summary['from_review']} reviewed)"
    )

    return Response(
        content=buffer.getvalue(),
        media_type="application/xml",
        headers={
            "Content-Disposition": 'attachment; filename="cvat_annotations.xml"',
            "X-Export-Count": str(summary["image_count"]),
            "X-Export-Reviewed": str(summary["from_review"]),
        },
    )


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> Response:
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        return Response(
            content="<h1>UI not built yet</h1><p>The API is up — try "
                    "<a href='/docs'>/docs</a>.</p>",
            media_type="text/html",
            status_code=503,
        )
    return FileResponse(page)


# Mounted unconditionally: the directory is created above, so a fresh
# checkout without the UI files still starts and serves the API.
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    print("\n  Road Annotator  ->  http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
