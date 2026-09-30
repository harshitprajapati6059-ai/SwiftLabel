"""
jobs.py - Runs pipeline work off the request thread and reports progress.

Inference takes ~11s per image on CPU, so annotating a folder cannot happen
inside an HTTP request: the browser would sit on a dead connection for
minutes with nothing to show. Instead a job is submitted, an id comes back
immediately, and the UI polls for progress.

EXACTLY ONE WORKER, deliberately. Three separate reasons converge on it:

  * SurfaceSegmenter is a process-wide singleton (utils.get_segmenter) and
    a torch module is not safe to call from several threads at once;
  * one inference pass already saturates the CPU, so a second worker would
    make both slower rather than either faster;
  * utils.config_overrides rebinds module globals for the duration of a
    run, which is only safe while runs cannot overlap.

If a second worker is ever added, all three have to be solved first.
"""

from __future__ import annotations

import itertools
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import config
import store
from utils import get_segmenter, logger, run_pipeline

# Only these may be overridden per run. The UI exposes exactly the three
# the README calls out as worth tuning; anything else has to be a config.py
# edit, so a stray key in a request body cannot quietly change the class
# table or repoint the model.
TUNABLE: dict[str, type] = {
    "MERGE_CLOSE_KERNEL_PX": int,
    "MIN_CLASS_PROBABILITY": float,
    "POLYGON_EPSILON_RATIO": float,
}

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
CANCELLED = "cancelled"

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_ids = itertools.count(1)

# Model load is ~14s and happens on first use. Tracked so the UI can show
# "warming up" instead of making the first annotate look pathologically slow.
_model_ready = threading.Event()
_model_error: Optional[str] = None


def clean_overrides(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Validate and coerce a client-supplied override map.

    Unknown keys are rejected outright rather than ignored: a typo'd knob
    that silently does nothing would look exactly like a knob that does not
    work, and the user would be left tuning a value with no effect.
    """
    if not raw:
        return {}

    unknown = sorted(set(raw) - set(TUNABLE))
    if unknown:
        raise ValueError(
            f"Not tunable from the UI: {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(TUNABLE))}."
        )

    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        caster = TUNABLE[key]
        try:
            cleaned[key] = caster(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be {caster.__name__}, got {value!r}") from None

    # An even closing kernel silently behaves like the odd size below it in
    # OpenCV, so the value that comes back would not be the value applied.
    kernel = cleaned.get("MERGE_CLOSE_KERNEL_PX")
    if kernel is not None:
        if kernel < 1:
            raise ValueError("MERGE_CLOSE_KERNEL_PX must be at least 1")
        if kernel % 2 == 0:
            cleaned["MERGE_CLOSE_KERNEL_PX"] = kernel + 1

    probability = cleaned.get("MIN_CLASS_PROBABILITY")
    if probability is not None and not 0.0 <= probability <= 1.0:
        raise ValueError("MIN_CLASS_PROBABILITY must be between 0 and 1")

    epsilon = cleaned.get("POLYGON_EPSILON_RATIO")
    if epsilon is not None and not 0.0 <= epsilon <= 1.0:
        raise ValueError("POLYGON_EPSILON_RATIO must be between 0 and 1")

    return cleaned


# ---------------------------------------------------------------------------
# Model warmup
# ---------------------------------------------------------------------------

def warm_up() -> None:
    """
    Queue the model load so it happens before any annotate job.

    Submitted to the same single-worker executor, so it cannot overlap with
    a run, and any job queued behind it simply waits for the weights.
    """
    def _load() -> None:
        global _model_error
        started = time.time()
        try:
            get_segmenter()
            logger.info(f"Model ready in {time.time() - started:.1f}s")
        except Exception as exc:
            _model_error = str(exc)
            logger.error(f"Model failed to load: {exc}")
        finally:
            _model_ready.set()

    _executor.submit(_load)


def model_status() -> dict[str, Any]:
    return {
        "ready": _model_ready.is_set() and _model_error is None,
        "loading": not _model_ready.is_set(),
        "error": _model_error,
        "name": config.MODEL_NAME,
    }


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def submit(image_names: list[str], overrides: Optional[dict[str, Any]] = None) -> str:
    """Queue an annotate run over the given images and return its id."""
    cleaned = clean_overrides(overrides)
    job_id = f"job-{next(_ids)}"

    record: dict[str, Any] = {
        "id": job_id,
        "state": QUEUED,
        "total": len(image_names),
        "done": 0,
        "current": None,
        "images": list(image_names),
        "succeeded": [],
        "errors": [],
        "overrides": cleaned,
        "started_at": None,
        "finished_at": None,
    }
    with _lock:
        _jobs[job_id] = record

    _executor.submit(_run_job, job_id, list(image_names), cleaned)
    return job_id


def _update(job_id: str, **fields: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def _run_job(job_id: str, image_names: list[str], overrides: dict[str, Any]) -> None:
    _update(job_id, state=RUNNING, started_at=time.time())

    for name in image_names:
        with _lock:
            job = _jobs.get(job_id)
            if job is None or job.get("cancel_requested"):
                _update(job_id, state=CANCELLED, current=None, finished_at=time.time())
                return

        _update(job_id, current=name)

        try:
            image_path = store.resolve_image(name)
            result = run_pipeline(image_path, overrides=overrides)

            # A re-run supersedes whatever the model said before, but not a
            # human's decision — an approved polygon stays exactly as the
            # reviewer left it until they explicitly revert it.
            human = store.human_annotation_path(image_path)
            superseded = human.exists()

            with _lock:
                job = _jobs.get(job_id)
                if job is not None:
                    job["succeeded"].append({
                        "name": name,
                        "regions": result.total_regions,
                        "seconds": result.processing_time_sec,
                        "warnings": result.warnings,
                        "kept_human_edits": superseded,
                    })
        except Exception as exc:
            logger.error(f"{name}: {exc}\n{traceback.format_exc()}")
            with _lock:
                job = _jobs.get(job_id)
                if job is not None:
                    job["errors"].append({"name": name, "error": str(exc)})

        with _lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["done"] += 1

    _update(job_id, state=DONE, current=None, finished_at=time.time())


def get(job_id: str) -> Optional[dict[str, Any]]:
    """Snapshot of a job's progress, or None if the id is unknown."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        snapshot = dict(job)

    elapsed = (
        (snapshot["finished_at"] or time.time()) - snapshot["started_at"]
        if snapshot["started_at"] else 0.0
    )
    snapshot["elapsed_sec"] = round(elapsed, 1)

    # Remaining time from this run's own measured pace rather than a
    # constant — it shifts with image size and with the merge kernel.
    if snapshot["done"] and snapshot["state"] == RUNNING:
        per_image = elapsed / snapshot["done"]
        snapshot["eta_sec"] = round(per_image * (snapshot["total"] - snapshot["done"]), 1)
    else:
        snapshot["eta_sec"] = None

    return snapshot


def cancel(job_id: str) -> bool:
    """
    Ask a job to stop after the image currently in flight.

    A pass cannot be interrupted mid-inference without leaving half-written
    artefacts, so the flag is checked between images.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None or job["state"] in (DONE, CANCELLED):
            return False
        job["cancel_requested"] = True
        return True


def active_job() -> Optional[dict[str, Any]]:
    """The running or queued job, if any — used to restore UI state on reload."""
    with _lock:
        for job in reversed(list(_jobs.values())):
            if job["state"] in (QUEUED, RUNNING):
                return dict(job)
    return None


def shutdown() -> None:
    _executor.shutdown(wait=False, cancel_futures=True)
