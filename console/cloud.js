// The console's point-cloud viewer: WebGL2, no library. It reads the one PLY the scans come as,
// draws it as round points coloured by the camera or by height, and turns around it with the
// mouse: drag turns, right-drag or shift-drag moves, the wheel zooms, a double-click goes back
// home. Loaded by console.js only when the 3D view opens.

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

  let count = 0, mode = 0, size = 3, zr = [0, 1], home = null, cam = null, frame = 0;
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
      count = cloud.count;
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers[0]);
      gl.bufferData(gl.ARRAY_BUFFER, cloud.pos, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers[1]);
      gl.bufferData(gl.ARRAY_BUFFER, cloud.col, gl.STATIC_DRAW);
      // extent and heights from a sample: the 1st-99th percentiles, so stray points do not count
      const step = Math.max(1, Math.floor(count / 50000));
      const axes = [[], [], []];
      for (let n = 0; n < count; n += step) for (let k = 0; k < 3; k++) axes[k].push(cloud.pos[n * 3 + k]);
      const pct = (a, q) => a[Math.min(a.length - 1, Math.floor(q * a.length))];
      axes.forEach(a => a.sort((x, y) => x - y));
      const lo = axes.map(a => pct(a, 0.01)), hi = axes.map(a => pct(a, 0.99));
      const floor = pct(axes[2], 0.02);
      zr = [floor, floor + Math.max(0.45, pct(axes[2], 0.98) - floor)];   // the arena: 0-45 cm
      // home: from above at 54 degrees, looking across the short side, so the long one runs
      // across the screen
      const [sx, sy] = [Math.max(hi[0] - lo[0], 0.5), Math.max(hi[1] - lo[1], 0.5)];
      home = { target: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, floor + 0.1], el: 0.95,
        az: sx >= sy ? -Math.PI / 2 : Math.PI, across: Math.max(sx, sy), deep: Math.min(sx, sy) };
      api.reset();
      return { count, size: hi.map((h, k) => h - lo[k]) };
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
  return api;
}
