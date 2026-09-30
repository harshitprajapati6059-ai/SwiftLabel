/* editor.js — polygon editing on the canvas.

   Shapes are held as {label, points:[{x,y}]} in IMAGE pixel coordinates,
   matching the on-disk schema exactly, so saving is a straight handoff with
   no conversion step where a rounding bug could hide.

   All hit-testing happens in SCREEN space. A grab radius fixed in image
   pixels would shrink to nothing when zoomed out on a 1600px frame and
   become an enormous slab when zoomed in; a radius fixed on screen behaves
   the same at every zoom level, which is what a hand actually expects. */

const GRAB_PX = 9;         // pointer tolerance for vertices and edges
const VERTEX_R = 4.5;      // drawn radius of a vertex handle
const MIN_POINTS = 3;      // below this it is not a polygon

class PolygonEditor {
  constructor(canvas, stage, hooks = {}) {
    this.view = new Viewport(canvas);
    this.canvas = canvas;
    this.stage = stage;
    this.hooks = hooks;

    this.image = null;
    this.shapes = [];
    this.classes = new Map();
    this.defaultLabel = null;
    this.fillAlpha = 0.35;

    this.tool = 'select';
    this.selected = -1;
    this.hover = null;          // {shape, vertex} | {shape, edge, point}
    this.drag = null;
    this.draft = null;          // in-progress polygon
    this.spaceHeld = false;

    this.undoStack = [];
    this.redoStack = [];

    this._bind();
  }

  /* ---- setup ---------------------------------------------------------- */

  setClasses(classes, fillAlpha) {
    this.classes = new Map(classes.map(c => [c.key, c]));
    this.defaultLabel = classes.length ? classes[0].key : null;
    if (fillAlpha != null) this.fillAlpha = fillAlpha;
  }

  setImage(image) {
    this.image = image;
    this.view.setImage(image.naturalWidth, image.naturalHeight);
    this.view.resize();
    this.view.fit();
    this.render();
  }

  setShapes(shapes) {
    this.shapes = (shapes || []).map(s => ({
      label: s.label,
      points: s.points.map(p => ({ x: p.x, y: p.y })),
    }));
    this.selected = this.shapes.length ? 0 : -1;
    this.undoStack = [];
    this.redoStack = [];
    this.draft = null;
    this.render();
    this._notifySelection();
  }

  getShapes() {
    return this.shapes.map(s => ({
      label: s.label,
      points: s.points.map(p => ({ x: p.x, y: p.y })),
    }));
  }

  setTool(tool) {
    if (this.tool === 'draw' && tool !== 'draw') this.cancelDraft();
    this.tool = tool;
    this.stage.dataset.tool = tool;
    this.render();
  }

  /* ---- undo ----------------------------------------------------------- */

  /* Snapshots rather than inverse operations: a polygon is ~30 points, so a
     deep copy is trivially cheap, and it cannot drift out of sync with the
     live state the way a hand-written undo for each operation can. */
  _snapshot() {
    this.undoStack.push(JSON.stringify(this.shapes));
    if (this.undoStack.length > 50) this.undoStack.shift();
    this.redoStack.length = 0;
  }

  undo() { this._timeTravel(this.undoStack, this.redoStack); }
  redo() { this._timeTravel(this.redoStack, this.undoStack); }

  _timeTravel(from, to) {
    if (!from.length) return;
    to.push(JSON.stringify(this.shapes));
    this.shapes = JSON.parse(from.pop());
    if (this.selected >= this.shapes.length) this.selected = this.shapes.length - 1;
    this._changed();
  }

  canUndo() { return this.undoStack.length > 0; }
  canRedo() { return this.redoStack.length > 0; }

  _changed() {
    this.render();
    this._notifySelection();
    this.hooks.onChange?.();
  }

  _notifySelection() {
    this.hooks.onSelect?.(this.selected, this.shapes);
  }

  /* ---- geometry ------------------------------------------------------- */

  _grabRadius() { return GRAB_PX / this.view.scale; }

  _hitVertex(point) {
    const radius = this._grabRadius();
    // Reverse order so the topmost drawn shape wins the click.
    for (let s = this.shapes.length - 1; s >= 0; s--) {
      const pts = this.shapes[s].points;
      for (let v = 0; v < pts.length; v++) {
        if (Math.hypot(pts[v].x - point.x, pts[v].y - point.y) <= radius) {
          return { shape: s, vertex: v };
        }
      }
    }
    return null;
  }

  _hitEdge(point) {
    const radius = this._grabRadius();
    for (let s = this.shapes.length - 1; s >= 0; s--) {
      const pts = this.shapes[s].points;
      for (let i = 0; i < pts.length; i++) {
        const a = pts[i];
        const b = pts[(i + 1) % pts.length];
        const projected = projectOnSegment(point, a, b);
        if (Math.hypot(projected.x - point.x, projected.y - point.y) <= radius) {
          return { shape: s, edge: i, point: projected };
        }
      }
    }
    return null;
  }

  _hitInside(point) {
    for (let s = this.shapes.length - 1; s >= 0; s--) {
      if (pointInPolygon(point, this.shapes[s].points)) return s;
    }
    return -1;
  }

  /* ---- events --------------------------------------------------------- */

  _bind() {
    const c = this.canvas;
    c.addEventListener('pointerdown', e => this._onDown(e));
    c.addEventListener('pointermove', e => this._onMove(e));
    window.addEventListener('pointerup', e => this._onUp(e));
    c.addEventListener('wheel', e => this._onWheel(e), { passive: false });
    c.addEventListener('contextmenu', e => this._onContext(e));
    c.addEventListener('dblclick', e => this._onDoubleClick(e));
    window.addEventListener('resize', () => {
      this.view.resize();
      this.render();
    });
  }

  setSpace(held) {
    this.spaceHeld = held;
    this.stage.classList.toggle('is-panning', held && !!this.drag);
  }

  _onDown(event) {
    if (!this.image) return;
    this.canvas.setPointerCapture(event.pointerId);
    const screen = this.view.eventPoint(event);
    const point = this.view.toImage(screen.x, screen.y);

    const wantsPan =
      event.button === 1 || this.spaceHeld || this.tool === 'pan';
    if (wantsPan) {
      this.drag = { kind: 'pan', screen };
      this.stage.classList.add('is-panning');
      return;
    }
    if (event.button !== 0) return;

    if (this.tool === 'draw') {
      this._draftPoint(point);
      return;
    }

    const vertex = this._hitVertex(point);
    if (vertex) {
      this.selected = vertex.shape;
      this._snapshot();
      this.drag = { kind: 'vertex', ...vertex };
      this._notifySelection();
      this.render();
      return;
    }

    // Inserting on an edge immediately begins dragging the new vertex, so
    // "add a point here and put it there" is one gesture rather than two.
    const edge = this._hitEdge(point);
    if (edge) {
      this._snapshot();
      const pts = this.shapes[edge.shape].points;
      pts.splice(edge.edge + 1, 0, { x: edge.point.x, y: edge.point.y });
      this.selected = edge.shape;
      this.drag = { kind: 'vertex', shape: edge.shape, vertex: edge.edge + 1 };
      this._changed();
      return;
    }

    const inside = this._hitInside(point);
    if (inside >= 0) {
      this.selected = inside;
      this._snapshot();
      this.drag = { kind: 'shape', shape: inside, last: point, moved: false };
      this._notifySelection();
      this.render();
      return;
    }

    // Empty space: drop the selection and let the drag pan, which is far
    // more useful than a marquee on a workspace holding one polygon.
    this.selected = -1;
    this.drag = { kind: 'pan', screen };
    this.stage.classList.add('is-panning');
    this._notifySelection();
    this.render();
  }

  _onMove(event) {
    if (!this.image) return;
    const screen = this.view.eventPoint(event);
    const point = this.view.toImage(screen.x, screen.y);

    if (this.drag) {
      if (this.drag.kind === 'pan') {
        this.view.panBy(screen.x - this.drag.screen.x, screen.y - this.drag.screen.y);
        this.drag.screen = screen;
        this.render();
        return;
      }
      if (this.drag.kind === 'vertex') {
        const pts = this.shapes[this.drag.shape].points;
        pts[this.drag.vertex] = this._clamp(point);
        this.render();
        this.hooks.onChange?.();
        return;
      }
      if (this.drag.kind === 'shape') {
        const dx = point.x - this.drag.last.x;
        const dy = point.y - this.drag.last.y;
        for (const p of this.shapes[this.drag.shape].points) { p.x += dx; p.y += dy; }
        this.drag.last = point;
        this.drag.moved = true;
        this.render();
        this.hooks.onChange?.();
        return;
      }
    }

    if (this.tool === 'draw' && this.draft) {
      this.draft.cursor = point;
      this.render();
      return;
    }

    // Hover feedback: vertices take priority over edges, so a point sitting
    // on its own edge is still grabbable rather than inserting a duplicate.
    const previous = JSON.stringify(this.hover);
    this.hover = this.tool === 'select'
      ? (this._hitVertex(point) || this._hitEdge(point))
      : null;
    if (JSON.stringify(this.hover) !== previous) this.render();
  }

  _onUp() {
    if (!this.drag) return;
    const wasVertex = this.drag.kind === 'vertex';
    const movedShape = this.drag.kind === 'shape' && this.drag.moved;

    if (this.drag.kind === 'shape' && !this.drag.moved) {
      // A click that selected without moving pushed a snapshot it never
      // used; dropping it keeps Ctrl+Z from consuming a no-op step.
      this.undoStack.pop();
    }
    if (wasVertex || movedShape) {
      for (const shape of this.shapes) {
        shape.points = shape.points.map(p => this._clamp(p));
      }
    }

    this.drag = null;
    this.stage.classList.remove('is-panning');
    if (wasVertex || movedShape) this._changed();
  }

  _onWheel(event) {
    event.preventDefault();
    const screen = this.view.eventPoint(event);
    this.view.zoomAt(screen.x, screen.y, event.deltaY < 0 ? 1.12 : 1 / 1.12);
    this.render();
    this.hooks.onZoom?.(this.view.scale);
  }

  _onContext(event) {
    event.preventDefault();
    if (this.tool !== 'select' || !this.image) return;
    const screen = this.view.eventPoint(event);
    const hit = this._hitVertex(this.view.toImage(screen.x, screen.y));
    if (!hit) return;

    const pts = this.shapes[hit.shape].points;
    if (pts.length <= MIN_POINTS) {
      this.hooks.onMessage?.(
        `A polygon needs at least ${MIN_POINTS} points — delete the whole shape instead.`,
        'error');
      return;
    }
    this._snapshot();
    pts.splice(hit.vertex, 1);
    this.hover = null;
    this._changed();
  }

  _onDoubleClick(event) {
    if (this.tool === 'draw' && this.draft) {
      event.preventDefault();
      this.commitDraft();
    }
  }

  _clamp(point) {
    return {
      x: Math.min(Math.max(point.x, 0), this.view.imageWidth - 1),
      y: Math.min(Math.max(point.y, 0), this.view.imageHeight - 1),
    };
  }

  /* ---- drawing new polygons ------------------------------------------- */

  _draftPoint(point) {
    if (!this.draft) this.draft = { points: [], cursor: point };
    this.draft.points.push(this._clamp(point));
    this.render();
  }

  commitDraft() {
    if (!this.draft || this.draft.points.length < MIN_POINTS) {
      this.hooks.onMessage?.(`Need at least ${MIN_POINTS} points.`, 'error');
      return false;
    }
    this._snapshot();
    this.shapes.push({ label: this.defaultLabel, points: this.draft.points });
    this.selected = this.shapes.length - 1;
    this.draft = null;
    this._changed();
    return true;
  }

  cancelDraft() {
    if (!this.draft) return;
    this.draft = null;
    this.render();
  }

  get isDrafting() { return !!this.draft; }

  /* ---- mutations ------------------------------------------------------ */

  deleteSelected() {
    if (this.selected < 0) return false;
    this._snapshot();
    this.shapes.splice(this.selected, 1);
    this.selected = this.shapes.length ? 0 : -1;
    this._changed();
    return true;
  }

  select(index) {
    this.selected = index;
    this.render();
    this._notifySelection();
  }

  setLabel(index, label) {
    if (!this.shapes[index]) return;
    this._snapshot();
    this.shapes[index].label = label;
    this._changed();
  }

  fit() {
    this.view.resize();
    this.view.fit();
    this.render();
    this.hooks.onZoom?.(this.view.scale);
  }

  /* ---- rendering ------------------------------------------------------ */

  _colorFor(label) {
    return this.classes.get(label)?.color || '#7c5cff';
  }

  render() {
    const { view } = this;
    view.clear();
    if (!this.image) return;

    view.withTransform(ctx => {
      ctx.imageSmoothingEnabled = view.scale < 4;
      ctx.drawImage(this.image, 0, 0);
    });

    const stroke = 2 / view.scale;

    this.shapes.forEach((shape, index) => {
      if (shape.points.length < 2) return;
      const color = this._colorFor(shape.label);
      const isSelected = index === this.selected;

      view.withTransform(ctx => {
        ctx.beginPath();
        shape.points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
        ctx.closePath();

        ctx.fillStyle = hexToRgba(color, isSelected ? this.fillAlpha : this.fillAlpha * 0.62);
        ctx.fill();

        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? stroke * 1.5 : stroke;
        ctx.lineJoin = 'round';
        ctx.stroke();
      });

      if (isSelected) this._drawHandles(shape, color);
    });

    if (this.hover?.edge !== undefined) this._drawGhost(this.hover.point);
    if (this.draft) this._drawDraft();
  }

  /* Handles are drawn outside withTransform so their size is in screen
     pixels — a handle scaled with the image would be a single unclickable
     dot when zoomed out and cover half the frame when zoomed in. */
  _drawHandles(shape, color) {
    const { ctx } = this.view;
    const hoveredVertex =
      this.hover?.vertex !== undefined && this.shapes[this.hover.shape] === shape
        ? this.hover.vertex : -1;

    shape.points.forEach((point, index) => {
      const screen = this.view.toScreen(point.x, point.y);
      const dragging =
        this.drag?.kind === 'vertex' &&
        this.shapes[this.drag.shape] === shape &&
        this.drag.vertex === index;
      const active = index === hoveredVertex || dragging;
      const radius = active ? VERTEX_R + 2 : VERTEX_R;

      ctx.beginPath();
      ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = active ? color : '#ffffff';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = active ? '#ffffff' : 'rgba(0,0,0,.75)';
      ctx.stroke();
    });
  }

  _drawGhost(point) {
    const { ctx } = this.view;
    const screen = this.view.toScreen(point.x, point.y);
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, VERTEX_R, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,.55)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.strokeStyle = '#12141a';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(screen.x - 2.5, screen.y); ctx.lineTo(screen.x + 2.5, screen.y);
    ctx.moveTo(screen.x, screen.y - 2.5); ctx.lineTo(screen.x, screen.y + 2.5);
    ctx.stroke();
  }

  _drawDraft() {
    const { ctx } = this.view;
    const color = this._colorFor(this.defaultLabel);
    const pts = this.draft.points.map(p => this.view.toScreen(p.x, p.y));
    if (!pts.length) return;

    ctx.beginPath();
    pts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    if (this.draft.cursor) {
      const cursor = this.view.toScreen(this.draft.cursor.x, this.draft.cursor.y);
      ctx.lineTo(cursor.x, cursor.y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    pts.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, i === 0 ? VERTEX_R + 1.5 : VERTEX_R, 0, Math.PI * 2);
      ctx.fillStyle = i === 0 ? color : '#ffffff';
      ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,.7)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }
}

/* ---- geometry helpers -------------------------------------------------- */

function projectOnSegment(point, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSq = dx * dx + dy * dy;
  if (!lengthSq) return { x: a.x, y: a.y };
  let t = ((point.x - a.x) * dx + (point.y - a.y) * dy) / lengthSq;
  t = Math.max(0, Math.min(1, t));
  return { x: a.x + t * dx, y: a.y + t * dy };
}

/* Ray casting. Points exactly on an edge are not guaranteed either way,
   which is fine: an edge click is caught by _hitEdge before this runs. */
function pointInPolygon(point, points) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const a = points[i];
    const b = points[j];
    const straddles = (a.y > point.y) !== (b.y > point.y);
    if (straddles && point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x) {
      inside = !inside;
    }
  }
  return inside;
}

function hexToRgba(hex, alpha) {
  const value = parseInt(hex.slice(1), 16);
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}
