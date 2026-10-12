// The console's point-cloud viewer: WebGL2, no library. It reads the one PLY the scans come as,
// draws it as round points coloured by the camera or by height -- or a Gaussian splat (.splat), as
// its Gaussians -- and turns around it with the mouse: drag turns, right-drag or shift-drag moves,
// the wheel zooms, a double-click goes back home. Loaded by console.js only when the 3D view opens.

// What m3pro/arena/ortho.py's write_ply writes, and nothing else: binary little-endian, each
// point float x y z then uchar red green blue, 15 bytes.
const HEADER = new RegExp('^ply\nformat binary_little_endian 1\\.0\nelement vertex (\\d+)\n' +
  'property float x\nproperty float y\nproperty float z\n' +
  'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n');

export function parsePly(buffer) {
  const head = new TextDecoder('latin1').decode(new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 512)));
  const m = HEADER.exec(head);   // the header is ASCII: characters are bytes
  if (!m) throw new Error('not the PLY ortho.py writes (binary little-endian; float x y z, uchar red green blue)');
  const count = Number(m[1]);
  if (m[0].length + count * 15 > buffer.byteLength) throw new Error('the PLY is shorter than its header says');
  const dv = new DataView(buffer, m[0].length);
  const pos = new Float32Array(count * 3), col = new Uint8Array(count * 3);
  for (let n = 0; n < count; n++) {
    for (let k = 0; k < 3; k++) {
      pos[n * 3 + k] = dv.getFloat32(n * 15 + 4 * k, true);
      col[n * 3 + k] = dv.getUint8(n * 15 + 12 + k);
    }
  }
  return { pos, col, count };
}

const VERTEX = `#version 300 es
layout(location = 0) in vec3 p;
layout(location = 1) in vec3 c;
uniform mat4 mvp;
uniform float size;
uniform int mode;
uniform vec2 zr;
out vec3 colour;
vec3 turbo(float x) {   // Google's polynomial fit of the Turbo colour map
  const vec4 r4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 g4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 b4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 r2 = vec2(-152.94239396, 59.28637943);
  const vec2 g2 = vec2(4.27729857, 2.82956604);
  const vec2 b2 = vec2(-89.90310912, 27.34824973);
  x = clamp(x, 0.0, 1.0);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return vec3(dot(v4, r4) + dot(v2, r2), dot(v4, g4) + dot(v2, g2), dot(v4, b4) + dot(v2, b2));
}
void main() {
  gl_Position = mvp * vec4(p, 1.0);
  gl_PointSize = size;
  colour = mode == 0 ? c : turbo((p.z - zr.x) / (zr.y - zr.x));
}`;

const FRAGMENT = `#version 300 es
precision mediump float;
in vec3 colour;
out vec4 o;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  if (dot(d, d) > 0.25) discard;   // round points
  o = vec4(colour, 1.0);
}`;

// Gaussian splats (train/splat/train_splat.py's .splat, antimatter15's layout): 32 bytes a Gaussian --
// position (3 float32, m), size (3 float32, m), colour and opacity (4 uint8), rotation (4 uint8: w x
// y z, each q * 128 + 128). Each is drawn as a quad facing the camera, as wide as its 3D covariance
// projects (EWA: J W S J^T W^T), its colour fading as a Gaussian; all blended back to front, sorted
// again whenever the view changes. Its data in a float texture, 4 texels each.
export function parseSplat(buffer) {
  if (!buffer.byteLength || buffer.byteLength % 32) throw new Error('not a .splat (32 bytes a Gaussian)');
  const count = buffer.byteLength / 32;
  const f = new Float32Array(buffer), b = new Uint8Array(buffer);
  const pos = new Float32Array(count * 3);
  const width = SPLAT_TEX_W, rows = Math.ceil(count * 4 / width);
  const tex = new Float32Array(width * rows * 4);
  for (let n = 0; n < count; n++) {
    const [x, y, z, sx, sy, sz] = f.subarray(n * 8, n * 8 + 6);
    const c = b.subarray(n * 32 + 24, n * 32 + 28);
    let [w, qx, qy, qz] = [...b.subarray(n * 32 + 28, n * 32 + 32)].map(v => (v - 128) / 128);
    const qn = Math.hypot(w, qx, qy, qz) || 1;
    [w, qx, qy, qz] = [w / qn, qx / qn, qy / qn, qz / qn];
    const R = [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - w * qz), 2 * (qx * qz + w * qy),
      2 * (qx * qy + w * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - w * qx),
      2 * (qx * qz - w * qy), 2 * (qy * qz + w * qx), 1 - 2 * (qx * qx + qy * qy)];
    const s = [sx, sy, sz];
    const M = R.map((r, k) => r * s[k % 3]);   // R times the sizes along its columns
    const cov = (i, j) => M[i * 3] * M[j * 3] + M[i * 3 + 1] * M[j * 3 + 1] + M[i * 3 + 2] * M[j * 3 + 2];
    pos.set([x, y, z], n * 3);
    tex.set([x, y, z, c[0] / 255, c[1] / 255, c[2] / 255, c[3] / 255, cov(0, 0),
      cov(0, 1), cov(0, 2), cov(1, 1), cov(1, 2), cov(2, 2), 0, 0, 0], n * 16);
  }
  return { pos, tex, count, width, rows };
}

const SPLAT_TEX_W = 4096;

const SPLAT_VERTEX = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D data;   // 4 texels a Gaussian: x y z r | g b a c00 | c01 c02 c11 c12 | c22
uniform mat4 view, proj;
uniform vec2 focal, viewport;   // pixels
layout(location = 0) in vec2 corner;   // the quad, in standard deviations
layout(location = 1) in uint index;    // which Gaussian: back to front
out vec4 colour;
out vec2 at;
vec4 texel(int i) { int w = textureSize(data, 0).x; return texelFetch(data, ivec2(i % w, i / w), 0); }
void main() {
  int i = int(index) * 4;
  vec4 t0 = texel(i), t1 = texel(i + 1), t2 = texel(i + 2), t3 = texel(i + 3);
  vec4 cam = view * vec4(t0.xyz, 1.0);
  vec4 clip = proj * cam;
  float bound = 1.2 * clip.w;
  gl_Position = vec4(0.0, 0.0, 2.0, 1.0);   // culled, unless it is in view
  if (cam.z > -0.01 || abs(clip.x) > bound || abs(clip.y) > bound) return;
  mat3 sigma = mat3(t1.w, t2.x, t2.y, t2.x, t2.z, t2.w, t2.y, t2.w, t3.x);
  float d = -cam.z;
  mat3 J = mat3(focal.x / d, 0.0, 0.0, 0.0, focal.y / d, 0.0,
                focal.x * cam.x / (d * d), focal.y * cam.y / (d * d), 0.0);
  mat3 T = J * mat3(view);
  mat3 cov = T * sigma * transpose(T);
  float a = cov[0][0] + 0.3, b = cov[0][1], c = cov[1][1] + 0.3;   // + a pixel's blur
  float mid = 0.5 * (a + c), r = length(vec2(0.5 * (a - c), b));
  float l1 = mid + r, l2 = max(mid - r, 0.0);
  if (l1 <= 0.0) return;
  vec2 v1 = abs(b) < 1e-9 ? (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) : normalize(vec2(b, l1 - a));
  vec2 v2 = vec2(-v1.y, v1.x);
  vec2 offset = corner.x * sqrt(l1) * v1 + corner.y * sqrt(l2) * v2;   // pixels
  colour = vec4(t0.w, t1.xyz);
  at = corner;
  gl_Position = vec4(clip.xy / clip.w + 2.0 * offset / viewport, 0.0, 1.0);
}`;

const SPLAT_FRAGMENT = `#version 300 es
precision highp float;
in vec4 colour;
in vec2 at;
out vec4 o;
void main() {
  float r = dot(at, at);
  if (r > 9.0) discard;   // three standard deviations
  float alpha = colour.a * exp(-0.5 * r);
  if (alpha < 1.0 / 255.0) discard;
  o = vec4(colour.rgb * alpha, alpha);   // premultiplied: blended back to front
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

// 4 x 4 matrices, column-major as WebGL takes them
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit = a => { const n = Math.hypot(...a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
function lookAt(eye, target, up) {
  const z = unit(sub(eye, target)), x = unit(cross(up, z)), y = cross(z, x);
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
}

export function viewer(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: true });
  if (!gl) throw new Error('this browser has no WebGL2');
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const u = Object.fromEntries(['mvp', 'size', 'mode', 'zr'].map(n => [n, gl.getUniformLocation(program, n)]));
  const vao = gl.createVertexArray();
  const buffers = [gl.createBuffer(), gl.createBuffer()];
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers[0]);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers[1]);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.enable(gl.DEPTH_TEST);

  const splat = { program: gl.createProgram(), vao: gl.createVertexArray(), order: gl.createBuffer(),
    texture: gl.createTexture(), count: 0, pos: null, depth: null, keys: null, sorted: null, seen: '' };
  gl.attachShader(splat.program, compile(gl, gl.VERTEX_SHADER, SPLAT_VERTEX));
  gl.attachShader(splat.program, compile(gl, gl.FRAGMENT_SHADER, SPLAT_FRAGMENT));
  gl.linkProgram(splat.program);
  if (!gl.getProgramParameter(splat.program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(splat.program));
  const su = Object.fromEntries(['data', 'view', 'proj', 'focal', 'viewport']
    .map(n => [n, gl.getUniformLocation(splat.program, n)]));
  gl.bindVertexArray(splat.vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-3, -3, 3, -3, -3, 3, 3, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, splat.order);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
  gl.vertexAttribDivisor(1, 1);
  gl.bindVertexArray(null);

  function sortSplats(view) {   // back to front along the view, a counting sort on 16-bit depths
    const key = Array.from(view, v => v.toFixed(5)).join();
    if (key === splat.seen) return;
    splat.seen = key;
    const { pos, depth, keys, sorted, count: n } = splat;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const z = view[2] * pos[i * 3] + view[6] * pos[i * 3 + 1] + view[10] * pos[i * 3 + 2] + view[14];
      depth[i] = z;
      if (z < lo) lo = z;
      if (z > hi) hi = z;
    }
    const scale = 65535 / (hi - lo || 1), counts = new Uint32Array(65536);
    for (let i = 0; i < n; i++) counts[keys[i] = (depth[i] - lo) * scale | 0]++;   // the farthest: the lowest z
    for (let k = 1; k < 65536; k++) counts[k] += counts[k - 1];
    for (let i = n - 1; i >= 0; i--) sorted[--counts[keys[i]]] = i;
    gl.bindBuffer(gl.ARRAY_BUFFER, splat.order);
    gl.bufferData(gl.ARRAY_BUFFER, sorted, gl.DYNAMIC_DRAW);
  }

  let kind = 'cloud', count = 0, mode = 0, size = 3, zr = [0, 1], home = null, cam = null, frame = 0;
  const fov = 50 * Math.PI / 180;

  function draw() {
    frame = 0;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.055, 0.067, 0.086, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!count || !w || !h) return;
    if (!cam) return api.reset();   // loaded while hidden: home, now that the picture has a size
    const { target, az, el, dist } = cam;
    const eye = [target[0] + dist * Math.cos(el) * Math.cos(az), target[1] + dist * Math.cos(el) * Math.sin(az),
      target[2] + dist * Math.sin(el)];
    if (kind === 'splat') {
      const view = new Float32Array(lookAt(eye, target, [0, 0, 1]));
      sortSplats(view);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(splat.program);
      gl.uniformMatrix4fv(su.view, false, view);
      gl.uniformMatrix4fv(su.proj, false, new Float32Array(perspective(fov, w / h, dist / 200, dist * 20)));
      const focal = h / (2 * Math.tan(fov / 2));   // pixels, the same both ways
      gl.uniform2f(su.focal, focal, focal);
      gl.uniform2f(su.viewport, w, h);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, splat.texture);
      gl.uniform1i(su.data, 0);
      gl.bindVertexArray(splat.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, splat.count);
      return;
    }
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    const mvp = new DOMMatrix(perspective(fov, w / h, dist / 200, dist * 20))   // also column-major
      .multiply(new DOMMatrix(lookAt(eye, target, [0, 0, 1])));
    gl.useProgram(program);
    gl.uniformMatrix4fv(u.mvp, false, mvp.toFloat32Array());
    gl.uniform1f(u.size, size * dpr);
    gl.uniform1i(u.mode, mode);
    gl.uniform2f(u.zr, zr[0], zr[1]);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.POINTS, 0, count);
  }
  // Drawn at once when something is set (a hidden page gets no animation frames), and at most
  // once a frame while the mouse moves the view.
  const later = () => { if (!frame) frame = requestAnimationFrame(draw); };

  function move(dx, dy) {   // the target across the screen, as far as the pointer moved there
    const k = 2 * cam.dist * Math.tan(fov / 2) / canvas.clientHeight;
    const right = [-Math.sin(cam.az), Math.cos(cam.az), 0];
    const up = [-Math.sin(cam.el) * Math.cos(cam.az), -Math.sin(cam.el) * Math.sin(cam.az), Math.cos(cam.el)];
    for (let i = 0; i < 3; i++) cam.target[i] += k * (-dx * right[i] + dy * up[i]);
  }
  const zoom = factor => { cam.dist = Math.min(Math.max(cam.dist * factor, 0.05), 500); };

  const pointers = new Map();   // one: turn (or move); two, on a touch screen: pinch and move
  let pan = false, spread = 0;
  const apart = () => { const [a, b] = [...pointers.values()]; return Math.hypot(a.x - b.x, a.y - b.y); };
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('pointerdown', e => {
    if (!cam) return;
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    pan = e.button === 2 || e.shiftKey;
    if (pointers.size === 2) spread = apart();
    canvas.style.cursor = 'grabbing';
  });
  canvas.addEventListener('pointermove', e => {
    const last = pointers.get(e.pointerId);
    if (!last) return;
    const dx = e.clientX - last.x, dy = e.clientY - last.y;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
      const now = apart();
      if (spread && now) zoom(spread / now);
      spread = now;
      move(dx / 2, dy / 2);   // each finger moves half of the midpoint's way
    } else if (pan) {
      move(dx, dy);
    } else {
      cam.az -= dx * 0.008;
      cam.el = Math.min(Math.max(cam.el + dy * 0.008, -1.5), 1.5);
    }
    later();
  });
  const release = e => {
    pointers.delete(e.pointerId);
    if (!pointers.size) canvas.style.cursor = '';
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('wheel', e => {
    if (!cam) return;
    e.preventDefault();
    zoom(Math.exp(e.deltaY * 0.0012));
    later();
  }, { passive: false });
  canvas.addEventListener('dblclick', () => api.reset());
  const resize = new ResizeObserver(() => draw());
  resize.observe(canvas);

  const api = {
    load(buffer) {
      const cloud = parsePly(buffer);
      kind = 'cloud';
      count = cloud.count;
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers[0]);
      gl.bufferData(gl.ARRAY_BUFFER, cloud.pos, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers[1]);
      gl.bufferData(gl.ARRAY_BUFFER, cloud.col, gl.STATIC_DRAW);
      return fit(cloud.pos);
    },
    loadSplat(buffer) {
      const s = parseSplat(buffer);
      kind = 'splat';
      count = splat.count = s.count;
      Object.assign(splat, { pos: s.pos, depth: new Float32Array(s.count), keys: new Uint32Array(s.count),
        sorted: new Uint32Array(s.count), seen: '' });
      gl.bindTexture(gl.TEXTURE_2D, splat.texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, s.width, s.rows, 0, gl.RGBA, gl.FLOAT, s.tex);
      for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
      for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
      return fit(s.pos);
    },
    reset() {   // back home, at the distance that fits the scan into the picture as it is shaped now
      if (!home) return;
      if (!canvas.clientWidth || !canvas.clientHeight) { cam = null; return; }   // hidden: later
      const t = 2 * Math.tan(fov / 2), aspect = canvas.clientWidth / canvas.clientHeight || 1;
      // 1.45: room for the walls, and the near side is drawn larger than the far one
      const dist = 1.45 * Math.max(home.deep * Math.sin(home.el) / t, home.across / (t * aspect));
      cam = { target: [...home.target], az: home.az, el: home.el, dist };
      draw();
    },
    setColour(m) { mode = m === 'height' ? 1 : 0; draw(); },
    setSize(px) { size = px; draw(); },
    dispose() { resize.disconnect(); gl.getExtension('WEBGL_lose_context')?.loseContext(); },
  };

  function fit(pos) {   // the colour by height's range, and home, from what was loaded
    // extent and heights from a sample: the 1st-99th percentiles, so stray points do not count
    const step = Math.max(1, Math.floor(count / 50000));
    const axes = [[], [], []];
    for (let n = 0; n < count; n += step) for (let k = 0; k < 3; k++) axes[k].push(pos[n * 3 + k]);
    const pct = (a, q) => a[Math.min(a.length - 1, Math.floor(q * a.length))];
    axes.forEach(a => a.sort((x, y) => x - y));
    const lo = axes.map(a => pct(a, 0.01)), hi = axes.map(a => pct(a, 0.99));
    const floor = pct(axes[2], 0.02);
    zr = [floor, floor + Math.max(0.45, pct(axes[2], 0.98) - floor)];   // the arena: 0-45 cm
    // home: from above at 54 degrees, looking across the short side, so the long one runs
    // across the screen. A splat is one thing on its patch of floor: closer, the middle 80 % of it.
    const [a, b, least, up] = kind === 'splat' ? [0.1, 0.9, 0.25, 0.05] : [0.01, 0.99, 0.5, 0.1];
    const [l, r] = [axes.map(v => pct(v, a)), axes.map(v => pct(v, b))];
    const [sx, sy] = [Math.max(r[0] - l[0], least), Math.max(r[1] - l[1], least)];
    home = { target: [(l[0] + r[0]) / 2, (l[1] + r[1]) / 2, floor + up], el: 0.95,
      az: sx >= sy ? -Math.PI / 2 : Math.PI, across: Math.max(sx, sy), deep: Math.min(sx, sy) };
    api.reset();
    return { count, size: hi.map((h, k) => h - lo[k]) };
  }
  return api;
}
