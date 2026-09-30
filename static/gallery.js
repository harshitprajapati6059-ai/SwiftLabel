/* gallery.js — the thumbnail grid and the editor filmstrip.

   Both are pure renderers: they take the image list and a callback, and own
   no state of their own. Every tile points at /thumb rather than /file —
   the source frames are several hundred KB each, so a grid of a hundred
   would be tens of megabytes of decoding for images shown at 178px. */

const BADGE = {
  approved:    '✓',
  edited:      '✎',
  auto:        '●',
  unannotated: '',
};

const STATE_LABEL = {
  approved:    'Approved',
  edited:      'Edited',
  auto:        'Auto',
  unannotated: 'Pending',
};

const Gallery = {
  /* Cache-bust on mtime so a re-annotated image does not keep showing the
     browser's cached tile from before the run. */
  thumbSrc(image) {
    return `${API.thumbUrl(image.name)}?v=${Math.round(image.modified)}`;
  },

  renderGrid(container, images, filter, onOpen) {
    const visible = filter === 'all'
      ? images
      : images.filter(image => image.state === filter);

    container.replaceChildren();

    if (!visible.length) {
      container.append(this._empty(images.length, filter));
      return;
    }

    const fragment = document.createDocumentFragment();
    for (const image of visible) {
      fragment.append(this._tile(image, onOpen));
    }
    container.append(fragment);
  },

  _tile(image, onOpen) {
    const tile = document.createElement('div');
    tile.className = 'tile';
    tile.dataset.state = image.state;
    tile.title = `${image.name}\n${STATE_LABEL[image.state]}`;

    const img = document.createElement('img');
    img.src = this.thumbSrc(image);
    img.alt = image.name;
    img.loading = 'lazy';

    const badge = document.createElement('div');
    badge.className = 'tile-badge';
    badge.textContent = BADGE[image.state];

    const bar = document.createElement('div');
    bar.className = 'tile-bar';

    const name = document.createElement('span');
    name.className = 'tile-name';
    name.textContent = image.name;

    const count = document.createElement('span');
    count.className = 'tile-shapes';
    count.textContent = image.shape_count ? `${image.point_count} pts` : '—';

    bar.append(name, count);
    if (image.warnings?.length) {
      const warn = document.createElement('span');
      warn.className = 'tile-warn';
      warn.textContent = '⚠';
      warn.title = image.warnings.join('\n');
      bar.append(warn);
    }

    tile.append(img, badge, bar);
    tile.addEventListener('click', () => onOpen(image.name));
    return tile;
  },

  _empty(total, filter) {
    const box = document.createElement('div');
    box.className = 'empty';
    const title = document.createElement('b');

    if (!total) {
      title.textContent = 'No images yet';
      box.append(title, document.createTextNode(
        'Drop image files anywhere on this page, or use “Add images”.'));
    } else {
      title.textContent = `Nothing ${STATE_LABEL[filter]?.toLowerCase() || filter}`;
      box.append(title, document.createTextNode(
        `${total} image${total === 1 ? '' : 's'} in the project — switch the filter to see them.`));
    }
    return box;
  },

  renderFilmstrip(container, images, currentName, onOpen) {
    container.replaceChildren();
    const fragment = document.createDocumentFragment();

    for (const image of images) {
      const frame = document.createElement('div');
      frame.className = 'frame';
      frame.dataset.state = image.state;
      frame.title = image.name;
      if (image.name === currentName) {
        frame.classList.add('is-current');
        frame.dataset.current = 'true';
      }

      const img = document.createElement('img');
      img.src = this.thumbSrc(image);
      img.alt = '';
      img.loading = 'lazy';

      frame.append(img);
      frame.addEventListener('click', () => onOpen(image.name));
      fragment.append(frame);
    }

    container.append(fragment);

    // Keep the active frame in view when navigating with the arrow keys.
    const current = container.querySelector('[data-current="true"]');
    current?.scrollIntoView({ block: 'nearest', inline: 'center' });
  },
};
