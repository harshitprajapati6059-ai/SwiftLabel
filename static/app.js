/* app.js — application state and wiring.

   The loop this is built around: drop images in, auto-annotate the lot,
   walk the results correcting what the model got wrong, approve each one,
   export. Everything below serves that order. */

const $ = id => document.getElementById(id);

const state = {
  images: [],
  classes: [],
  defaults: {},          // config.py values for the tunable knobs
  tuning: {},            // session overrides, never written to config.py
  current: null,
  filter: 'all',
  dirty: false,
  saving: false,
  jobId: null,
  jobTimer: null,
};

let editor;
let saveTimer = null;

/* ── boot ─────────────────────────────────────────────────────────────── */

async function boot() {
  editor = new PolygonEditor($('canvas'), $('stage'), {
    onChange: onEditorChange,
    onSelect: renderShapeList,
    onZoom: scale => { $('zoom-badge').textContent = `${Math.round(scale * 100)}%`; },
    onMessage: toast,
  });

  try {
    const [classes, settings] = await Promise.all([API.classes(), API.settings()]);
    state.classes = classes;
    state.defaults = settings.tunable;
    state.tuning = { ...settings.tunable };
    editor.setClasses(classes, settings.overlay_alpha);
    renderClassList();
    initTuning();
  } catch (error) {
    toast(`Could not reach the server: ${error.message}`, 'error');
    return;
  }

  wireGallery();
  wireEditor();
  wireShortcuts();
  wireDropzone();

  await refreshImages();
  pollStatus();
  setInterval(pollStatus, 4000);
}

/* ── data ─────────────────────────────────────────────────────────────── */

async function refreshImages() {
  state.images = await API.images();
  renderGallery();
  renderCounts();
  if (document.documentElement.dataset.view === 'editor') {
    Gallery.renderFilmstrip($('filmstrip'), state.images, state.current, openImage);
    updateApproveButton();
  }
}

function imageRecord(name) {
  return state.images.find(image => image.name === name);
}

function renderCounts() {
  const tally = { approved: 0, edited: 0, auto: 0, unannotated: 0 };
  for (const image of state.images) tally[image.state]++;
  for (const [key, value] of Object.entries(tally)) $(`c-${key}`).textContent = value;

  const done = tally.approved;
  $('gallery-hint').textContent = state.images.length
    ? `${done} of ${state.images.length} approved`
    : '';
}

async function pollStatus() {
  try {
    const status = await API.status();
    const box = $('model-state');
    const model = status.model;
    box.dataset.ready = model.error ? 'error' : String(model.ready);
    $('model-label').textContent =
      model.error ? 'model failed' : model.ready ? 'model ready' : 'loading model…';
    box.title = model.error || model.name;

    // Resume a run that was started before a page reload.
    if (status.active_job && !state.jobId) attachJob(status.active_job.id);
  } catch { /* server restarting; the next tick will pick it up */ }
}

/* ── gallery ──────────────────────────────────────────────────────────── */

function renderGallery() {
  Gallery.renderGrid($('grid'), state.images, state.filter, openImage);
}

function wireGallery() {
  $('filters').addEventListener('click', event => {
    const button = event.target.closest('.filter');
    if (!button) return;
    state.filter = button.dataset.filter;
    for (const el of $('filters').children) el.classList.toggle('is-active', el === button);
    renderGallery();
  });

  $('btn-upload').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', event => {
    if (event.target.files.length) uploadFiles(event.target.files);
    event.target.value = '';
  });

  $('btn-annotate').addEventListener('click', annotateAll);
  $('btn-cancel-job').addEventListener('click', cancelJob);
  $('btn-export').addEventListener('click', openExportModal);
  $('btn-export-cancel').addEventListener('click', () => { $('export-modal').hidden = true; });
  $('btn-export-go').addEventListener('click', runExport);
}

function wireDropzone() {
  const zone = $('dropzone');
  let depth = 0;

  window.addEventListener('dragenter', event => {
    if (![...event.dataTransfer.types].includes('Files')) return;
    depth++;
    zone.classList.add('is-active');
  });
  window.addEventListener('dragleave', () => {
    if (--depth <= 0) { depth = 0; zone.classList.remove('is-active'); }
  });
  window.addEventListener('dragover', event => event.preventDefault());
  window.addEventListener('drop', event => {
    event.preventDefault();
    depth = 0;
    zone.classList.remove('is-active');
    if (event.dataTransfer.files.length) uploadFiles(event.dataTransfer.files);
  });
}

async function uploadFiles(files) {
  try {
    const result = await API.upload(files);
    await refreshImages();
    if (result.saved.length) {
      toast(`Added ${result.saved.length} image${result.saved.length === 1 ? '' : 's'}.`, 'success');
    }
    for (const bad of result.rejected) toast(`${bad.name}: ${bad.reason}`, 'error');
  } catch (error) {
    toast(error.message, 'error');
  }
}

/* ── jobs ─────────────────────────────────────────────────────────────── */

async function annotateAll() {
  try {
    const { job_id, images } = await API.startJob({ images: null, overrides: changedTuning() });
    toast(`Annotating ${images.length} image${images.length === 1 ? '' : 's'}…`);
    attachJob(job_id);
  } catch (error) {
    // 409 here means every image already has an annotation, which is a
    // normal end state rather than a failure.
    toast(error.message, error.message === 'Nothing to annotate' ? 'info' : 'error');
  }
}

function attachJob(jobId) {
  state.jobId = jobId;
  $('job-bar').hidden = false;
  $('btn-annotate').disabled = true;
  clearInterval(state.jobTimer);
  state.jobTimer = setInterval(tickJob, 600);
  tickJob();
}

async function tickJob() {
  let job;
  try {
    job = await API.job(state.jobId);
  } catch {
    return finishJob();
  }

  const pct = job.total ? (job.done / job.total) * 100 : 0;
  $('job-fill').style.width = `${pct}%`;

  const eta = job.eta_sec != null && job.eta_sec > 0 ? ` · ~${Math.ceil(job.eta_sec)}s left` : '';
  $('job-text').textContent = job.state === 'running'
    ? `${job.done}/${job.total}${job.current ? ` · ${job.current}` : ''}${eta}`
    : `${job.state} · ${job.done}/${job.total}`;

  if (job.state === 'done' || job.state === 'cancelled') {
    const failed = job.errors.length;
    if (failed) {
      for (const err of job.errors.slice(0, 3)) toast(`${err.name}: ${err.error}`, 'error');
    }
    // An image whose human copy already existed keeps it — the model just
    // re-ran underneath. Say so rather than letting it look like a no-op.
    const kept = job.succeeded.filter(s => s.kept_human_edits).length;
    if (kept) {
      toast(`${kept} image${kept === 1 ? '' : 's'} kept your saved annotation. ` +
            `Use “Revert to AI” to take the new one.`);
    }
    if (job.state === 'done' && !failed) {
      toast(`Annotated ${job.succeeded.length} image${job.succeeded.length === 1 ? '' : 's'}.`,
            'success');
    }
    finishJob();
    await refreshImages();
    if (state.current) await loadAnnotation(state.current);
  }
}

function finishJob() {
  clearInterval(state.jobTimer);
  state.jobTimer = null;
  state.jobId = null;
  $('job-bar').hidden = true;
  $('btn-annotate').disabled = false;
}

async function cancelJob() {
  if (state.jobId) await API.cancelJob(state.jobId);
}

/* ── editor ───────────────────────────────────────────────────────────── */

function wireEditor() {
  $('btn-back').addEventListener('click', closeEditor);
  $('btn-approve').addEventListener('click', approveCurrent);
  $('btn-revert').addEventListener('click', revertCurrent);
  $('btn-rerun').addEventListener('click', rerunCurrent);
  $('btn-fit').addEventListener('click', () => editor.fit());
  $('btn-undo').addEventListener('click', () => editor.undo());
  $('btn-redo').addEventListener('click', () => editor.redo());

  $('rail').addEventListener('click', event => {
    const button = event.target.closest('.tool[data-tool]');
    if (button) selectTool(button.dataset.tool);
  });
}

function selectTool(tool) {
  editor.setTool(tool);
  for (const button of document.querySelectorAll('.tool[data-tool]')) {
    button.classList.toggle('is-active', button.dataset.tool === tool);
  }
  $('draw-hint').hidden = tool !== 'draw';
}

async function openImage(name) {
  await flushSave();

  document.documentElement.dataset.view = 'editor';
  state.current = name;

  const record = imageRecord(name);
  $('editor-title').textContent = name;
  const index = state.images.findIndex(image => image.name === name);
  $('editor-counter').textContent = `${index + 1} of ${state.images.length}`;

  setBusy(true);
  try {
    const [image] = await Promise.all([
      loadImageElement(API.imageUrl(name)),
      loadAnnotation(name),
    ]);
    editor.setImage(image);
    selectTool('select');
    $('zoom-badge').textContent = `${Math.round(editor.view.scale * 100)}%`;
  } catch (error) {
    toast(`Could not open ${name}: ${error.message}`, 'error');
  } finally {
    setBusy(false);
  }

  renderWarnings(record);
  updateApproveButton();
  Gallery.renderFilmstrip($('filmstrip'), state.images, name, openImage);
}

function loadImageElement(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('image failed to decode'));
    image.src = url;
  });
}

async function loadAnnotation(name) {
  const payload = await API.annotation(name);
  editor.setShapes(payload.annotation?.shapes || []);
  setDirty(false);
  return payload;
}

async function closeEditor() {
  await flushSave();
  document.documentElement.dataset.view = 'gallery';
  state.current = null;
  await refreshImages();
}

function onEditorChange() {
  setDirty(true);
  renderShapeList(editor.selected, editor.shapes);
  $('btn-undo').disabled = !editor.canUndo();
  $('btn-redo').disabled = !editor.canRedo();

  // Debounced so a vertex drag is one request at the end rather than one
  // per mouse-move; flushSave() covers navigating away before it fires.
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveCurrent, 700);
}

function setDirty(dirty) {
  state.dirty = dirty;
  const label = $('save-state');
  label.dataset.dirty = state.saving ? 'saving' : String(dirty);
  label.textContent = state.saving ? 'saving…' : dirty ? 'unsaved' : 'saved';
}

async function saveCurrent() {
  if (!state.current || !state.dirty || state.saving) return;
  state.saving = true;
  setDirty(true);
  try {
    const result = await API.save(state.current, {
      shapes: editor.getShapes(),
      width: editor.view.imageWidth,
      height: editor.view.imageHeight,
    });
    state.saving = false;
    setDirty(false);
    const record = imageRecord(state.current);
    if (record) {
      record.state = result.state;
      record.shape_count = result.annotation.shapes.length;
      record.point_count = result.annotation.shapes
        .reduce((sum, shape) => sum + shape.points.length, 0);
    }
    updateApproveButton();
    Gallery.renderFilmstrip($('filmstrip'), state.images, state.current, openImage);
    renderCounts();
  } catch (error) {
    state.saving = false;
    setDirty(true);
    toast(`Save failed: ${error.message}`, 'error');
  }
}

/* Force any pending debounced save to complete before leaving an image. */
async function flushSave() {
  clearTimeout(saveTimer);
  if (state.dirty) await saveCurrent();
}

async function approveCurrent() {
  if (!state.current) return;
  await flushSave();
  const name = state.current;
  try {
    const record = imageRecord(name);
    if (record?.state === 'approved') {
      await API.unapprove(name);
      await refreshImages();
      toast('Approval withdrawn.');
      return;
    }
    await API.approve(name);
    await refreshImages();
    if (!(await goToNextPending())) {
      toast('All images approved — ready to export.', 'success');
    }
  } catch (error) {
    toast(error.message, 'error');
  }
}

/* Advance to the next image still needing review, wrapping around, so a
   long folder can be worked through without touching the filmstrip. */
async function goToNextPending() {
  const start = state.images.findIndex(image => image.name === state.current);
  for (let step = 1; step <= state.images.length; step++) {
    const candidate = state.images[(start + step) % state.images.length];
    if (candidate.state !== 'approved' && candidate.state !== 'unannotated') {
      await openImage(candidate.name);
      return true;
    }
  }
  return false;
}

async function revertCurrent() {
  if (!state.current) return;
  const record = imageRecord(state.current);
  if (record?.state === 'auto' || record?.state === 'unannotated') {
    toast('Nothing to revert — this is still the model’s own output.');
    return;
  }
  if (!confirm('Discard your edits and approval for this image?')) return;

  clearTimeout(saveTimer);
  state.dirty = false;
  try {
    await API.revert(state.current);
    await loadAnnotation(state.current);
    await refreshImages();
    toast('Restored the model’s polygon.');
  } catch (error) {
    toast(error.message, 'error');
  }
}

async function rerunCurrent() {
  if (!state.current) return;
  await flushSave();
  try {
    const { job_id } = await API.startJob({
      images: [state.current],
      overrides: changedTuning(),
    });
    attachJob(job_id);
  } catch (error) {
    toast(error.message, 'error');
  }
}

function setBusy(busy) {
  $('stage-overlay').classList.toggle('is-busy', busy);
  $('stage-overlay').textContent = busy ? 'Loading…' : '';
}

function updateApproveButton() {
  const record = imageRecord(state.current);
  const approved = record?.state === 'approved';
  const button = $('btn-approve');
  button.dataset.approved = String(approved);
  button.innerHTML = approved
    ? '✓ Approved <kbd>A</kbd>'
    : '✓ Approve <kbd>A</kbd>';
  button.disabled = !record || record.state === 'unannotated';
  button.title = record?.state === 'unannotated'
    ? 'Nothing to approve yet — annotate or draw a polygon first'
    : approved ? 'Click to withdraw approval' : 'Approve and go to the next image';
}

/* ── panels ───────────────────────────────────────────────────────────── */

function renderClassList() {
  const list = $('class-list');
  list.replaceChildren();
  for (const item of state.classes) {
    const row = document.createElement('div');
    row.className = 'class-row';
    row.title = item.drivable ? 'Drivable surface' : 'Not drivable';

    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = item.color;

    const label = document.createElement('span');
    label.textContent = item.key;

    const meta = document.createElement('small');
    meta.textContent = item.drivable ? 'drivable' : '—';

    row.append(swatch, label, meta);
    row.addEventListener('click', () => {
      if (editor.selected >= 0) editor.setLabel(editor.selected, item.key);
    });
    list.append(row);
  }
}

function renderShapeList(selected, shapes) {
  const list = $('shape-list');
  $('shape-count').textContent = shapes.length;
  list.replaceChildren();

  if (!shapes.length) {
    const empty = document.createElement('div');
    empty.className = 'list-empty';
    empty.textContent = 'No polygons. Press N to draw one.';
    list.append(empty);
    return;
  }

  shapes.forEach((shape, index) => {
    const row = document.createElement('div');
    row.className = 'shape-row';
    if (index === selected) row.classList.add('is-selected');

    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = state.classes.find(c => c.key === shape.label)?.color || '#7c5cff';

    const label = document.createElement('span');
    label.textContent = shape.label;

    const points = document.createElement('small');
    points.textContent = `${shape.points.length} pts`;

    const remove = document.createElement('button');
    remove.className = 'shape-del';
    remove.textContent = '×';
    remove.title = 'Delete this polygon';
    remove.addEventListener('click', event => {
      event.stopPropagation();
      editor.select(index);
      editor.deleteSelected();
    });

    row.append(swatch, label, points, remove);
    row.addEventListener('click', () => editor.select(index));
    list.append(row);
  });
}

function renderWarnings(record) {
  const block = $('warn-block');
  const list = $('warn-list');
  const warnings = record?.warnings || [];
  block.hidden = !warnings.length;
  list.replaceChildren();
  for (const warning of warnings) {
    const item = document.createElement('li');
    item.textContent = warning;
    list.append(item);
  }
}

/* ── tuning ───────────────────────────────────────────────────────────── */

function initTuning() {
  for (const block of document.querySelectorAll('.tune')) {
    const key = block.dataset.key;
    const slider = block.querySelector('input');
    slider.value = state.defaults[key];
    paintTuning(block, key);

    slider.addEventListener('input', () => {
      state.tuning[key] = Number(slider.value);
      paintTuning(block, key);
    });
  }
  $('btn-reset-tune').addEventListener('click', () => {
    state.tuning = { ...state.defaults };
    for (const block of document.querySelectorAll('.tune')) {
      block.querySelector('input').value = state.defaults[block.dataset.key];
      paintTuning(block, block.dataset.key);
    }
  });
}

function paintTuning(block, key) {
  const value = state.tuning[key];
  block.querySelector('.tune-val').textContent = value;
  block.classList.toggle('is-changed', value !== state.defaults[key]);
}

/* Only send what actually differs, so a default run records no overrides
   and its metadata stays honest about having used config.py's values. */
function changedTuning() {
  const changed = {};
  for (const [key, value] of Object.entries(state.tuning)) {
    if (value !== state.defaults[key]) changed[key] = value;
  }
  return changed;
}

/* ── export ───────────────────────────────────────────────────────────── */

async function openExportModal() {
  try {
    const preview = await API.exportPreview();
    $('x-approved').textContent = preview.approved;
    $('x-annotated').textContent = preview.annotated;

    const pending = preview.annotated - preview.approved;
    $('x-note').textContent = pending
      ? `${pending} annotated image${pending === 1 ? '' : 's'} not yet approved.`
      : '';
    $('export-modal').hidden = false;
  } catch (error) {
    toast(error.message, 'error');
  }
}

async function runExport() {
  const scope = document.querySelector('input[name="scope"]:checked').value;
  try {
    const { blob, count, reviewed } = await API.exportCvat(scope);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'cvat_annotations.xml';
    link.click();
    URL.revokeObjectURL(url);

    $('export-modal').hidden = true;
    toast(`Exported ${count} image${count === 1 ? '' : 's'} (${reviewed} reviewed).`, 'success');
  } catch (error) {
    toast(error.message, 'error');
  }
}

/* ── shortcuts ────────────────────────────────────────────────────────── */

function wireShortcuts() {
  window.addEventListener('keydown', async event => {
    // Never swallow typing, and never fight a native shortcut in a field.
    const tag = event.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || event.target.isContentEditable) return;

    if (event.key === 'Escape') {
      if (!$('export-modal').hidden) { $('export-modal').hidden = true; return; }
      if (editor?.isDrafting) { editor.cancelDraft(); return; }
    }

    if (document.documentElement.dataset.view !== 'editor') return;

    const ctrl = event.ctrlKey || event.metaKey;
    if (ctrl && event.key.toLowerCase() === 'z') {
      event.preventDefault();
      event.shiftKey ? editor.redo() : editor.undo();
      return;
    }
    if (ctrl && event.key.toLowerCase() === 'y') {
      event.preventDefault();
      editor.redo();
      return;
    }
    if (ctrl) return;

    switch (event.key) {
      case ' ':
        event.preventDefault();
        editor.setSpace(true);
        break;
      case 'Enter':
        if (editor.isDrafting && editor.commitDraft()) selectTool('select');
        break;
      case 'Delete':
      case 'Backspace':
        event.preventDefault();
        editor.deleteSelected();
        break;
      case 'ArrowLeft':  event.preventDefault(); await step(-1); break;
      case 'ArrowRight': event.preventDefault(); await step(1);  break;
      default:
        switch (event.key.toLowerCase()) {
          case 'a': await approveCurrent(); break;
          case 'n': selectTool('draw'); break;
          case 'v': selectTool('select'); break;
          case 'h': selectTool('pan'); break;
          case 'f': editor.fit(); break;
        }
    }
  });

  window.addEventListener('keyup', event => {
    if (event.key === ' ') editor?.setSpace(false);
  });

  // Last line of defence for an unsaved edit — a debounced save that has
  // not fired yet would otherwise be lost on a reload or a closed tab.
  window.addEventListener('beforeunload', event => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = '';
  });
}

async function step(delta) {
  const index = state.images.findIndex(image => image.name === state.current);
  if (index < 0) return;
  const next = state.images[(index + delta + state.images.length) % state.images.length];
  await openImage(next.name);
}

/* ── toasts ───────────────────────────────────────────────────────────── */

function toast(message, kind = 'info') {
  const element = document.createElement('div');
  element.className = 'toast';
  element.dataset.kind = kind;
  element.textContent = message;
  $('toasts').append(element);
  setTimeout(() => element.remove(), kind === 'error' ? 6500 : 3400);
}

boot();
