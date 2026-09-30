/* canvas.js — the viewport transform.

   Polygons are stored in IMAGE pixels (a 1600x1600 frame), but every mouse
   event arrives in SCREEN pixels, and the two only line up at 100% zoom.
   Everything that converts between the two lives here so the editor can
   work purely in image space and never has to think about the current zoom.

   Hit-testing is the exception and deliberately so: a grab radius has to be
   constant on screen, not in the image, or vertices would become impossible
   to grab when zoomed out. The editor converts its tolerance through
   `scale` for that reason. */

class Viewport {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.scale = 1;
    this.tx = 0;
    this.ty = 0;
    this.imageWidth = 0;
    this.imageHeight = 0;
    this.minScale = 0.02;
    this.maxScale = 40;
  }

  /* Match the backing store to the CSS size and the device pixel ratio, so
     lines stay crisp on a HiDPI display instead of being scaled up blurry. */
  resize() {
    const ratio = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.cssWidth = rect.width;
    this.cssHeight = rect.height;
    this.canvas.width = Math.round(rect.width * ratio);
    this.canvas.height = Math.round(rect.height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  setImage(width, height) {
    this.imageWidth = width;
    this.imageHeight = height;
  }

  fit(padding = 28) {
    if (!this.imageWidth || !this.cssWidth) return;
    const scale = Math.min(
      (this.cssWidth - padding * 2) / this.imageWidth,
      (this.cssHeight - padding * 2) / this.imageHeight,
    );
    this.scale = Math.max(this.minScale, Math.min(scale, this.maxScale));
    this.tx = (this.cssWidth - this.imageWidth * this.scale) / 2;
    this.ty = (this.cssHeight - this.imageHeight * this.scale) / 2;
  }

  toScreen(x, y) {
    return { x: x * this.scale + this.tx, y: y * this.scale + this.ty };
  }

  toImage(x, y) {
    return { x: (x - this.tx) / this.scale, y: (y - this.ty) / this.scale };
  }

  /* Screen coordinates relative to the canvas, from a raw pointer event. */
  eventPoint(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  /* Zoom about a fixed screen point, so the pixel under the cursor stays
     under the cursor — the behaviour every map and image tool has, and the
     reason this cannot just multiply `scale`. */
  zoomAt(screenX, screenY, factor) {
    const next = Math.max(this.minScale, Math.min(this.scale * factor, this.maxScale));
    if (next === this.scale) return;
    const before = this.toImage(screenX, screenY);
    this.scale = next;
    this.tx = screenX - before.x * this.scale;
    this.ty = screenY - before.y * this.scale;
  }

  panBy(dx, dy) {
    this.tx += dx;
    this.ty += dy;
  }

  clear() {
    this.ctx.clearRect(0, 0, this.cssWidth, this.cssHeight);
  }

  /* Run a callback with the transform applied, so image-space drawing needs
     no per-point arithmetic. Line widths are divided by scale by the caller
     where a constant on-screen thickness is wanted. */
  withTransform(draw) {
    const { ctx } = this;
    ctx.save();
    ctx.translate(this.tx, this.ty);
    ctx.scale(this.scale, this.scale);
    draw(ctx);
    ctx.restore();
  }
}
