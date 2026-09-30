/* api.js — thin wrapper over the backend.
   Image names go in the path and routinely contain spaces and parentheses,
   so every one of them is encoded here rather than at each call site. */

const API = (() => {
  async function request(method, url, body) {
    const options = { method, headers: {} };
    if (body !== undefined) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);
    if (!response.ok) {
      // FastAPI puts the useful message in `detail`; fall back to the status
      // text so a proxy error or a crash still surfaces something readable.
      let detail = response.statusText;
      try {
        const payload = await response.json();
        detail = payload.detail || detail;
        if (Array.isArray(detail)) detail = detail.map(d => d.msg || d).join('; ');
      } catch { /* not JSON */ }
      throw new Error(detail);
    }
    if (response.status === 204) return null;
    return response.json();
  }

  const enc = name => encodeURIComponent(name);

  return {
    status:      ()            => request('GET', '/api/status'),
    classes:     ()            => request('GET', '/api/classes'),
    settings:    ()            => request('GET', '/api/settings'),
    images:      ()            => request('GET', '/api/images'),

    imageUrl:    name          => `/api/images/${enc(name)}/file`,
    thumbUrl:    name          => `/api/images/${enc(name)}/thumb`,

    annotation:  name          => request('GET',    `/api/images/${enc(name)}/annotation`),
    save:        (name, body)  => request('PUT',    `/api/images/${enc(name)}/annotation`, body),
    revert:      name          => request('DELETE', `/api/images/${enc(name)}/annotation`),
    approve:     name          => request('POST',   `/api/images/${enc(name)}/approve`),
    unapprove:   name          => request('DELETE', `/api/images/${enc(name)}/approve`),
    remove:      name          => request('DELETE', `/api/images/${enc(name)}`),

    startJob:    body          => request('POST', '/api/jobs', body),
    job:         id            => request('GET',  `/api/jobs/${id}`),
    cancelJob:   id            => request('POST', `/api/jobs/${id}/cancel`),

    exportPreview: ()          => request('GET', '/api/export/preview'),

    async upload(files) {
      const form = new FormData();
      for (const file of files) form.append('files', file);
      const response = await fetch('/api/upload', { method: 'POST', body: form });
      if (!response.ok) throw new Error((await response.json()).detail || 'Upload failed');
      return response.json();
    },

    /* Returned as a blob rather than JSON so the browser can save the XML
       straight to disk — the same bytes the CLI writes to output/. */
    async exportCvat(scope) {
      const response = await fetch('/api/export/cvat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scope }),
      });
      if (!response.ok) throw new Error((await response.json()).detail || 'Export failed');
      return {
        blob: await response.blob(),
        count: Number(response.headers.get('X-Export-Count') || 0),
        reviewed: Number(response.headers.get('X-Export-Reviewed') || 0),
      };
    },
  };
})();
