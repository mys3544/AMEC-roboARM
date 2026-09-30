'use strict';
// The console page: every robot in robots.json is polled all the time (so STOP ALL and the
// status dots cover robots whose card is hidden); a card per robot shown. JSON goes through
// serve.py as /robot/<id>/<path>; pictures come straight from the robot.

const $ = (sel, root = document) => root.querySelector(sel);
const DIRS = { w: [1, 0, 0], s: [-1, 0, 0], a: [0, 1, 0], d: [0, -1, 0], q: [0, 0, 1], e: [0, 0, -1] };
const STOPS = {   // what STOP means for each kind of robot
  m3: robot => call(robot, 'api/stop', {}),
  x3: robot => Promise.all([call(robot, 'api/auto/stop', {}), call(robot, 'api/arm/hold', {})]),
};
const store = {   // per-viewer conveniences only; the page works without them
  get(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ } },
};

function log(text, error = false) {
  const line = document.createElement('div');
  line.textContent = `${new Date().toLocaleTimeString()}  ${text}`;
  if (error) line.className = 'err';
  $('#log').prepend(line);
  while ($('#log').childElementCount > 100) $('#log').lastElementChild.remove();
}

async function call(robot, path, body) {
  const options = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  };
  const reply = await fetch(`/robot/${robot.id}/${path}`, options);
  const data = await reply.json().catch(() => ({}));
  if (!reply.ok || data.ok === false) throw new Error(data.error || `HTTP ${reply.status}`);
  return data;
}

class Robot {
  constructor(spec) {
    Object.assign(this, spec);
    this.state = null;
    this.up = null;
    this.card = null;
    this.dot = document.createElement('span');
    this.dot.innerHTML = `<span class="dot"></span>${spec.name}`;
    $('#links').append(this.dot);
    this.poll();
  }

  async poll() {
    try {
      this.state = await call(this, 'api/state');
      this.setUp(true);
    } catch (e) {
      this.state = null;
      this.error = e.message;
      this.setUp(false);
    }
    this.card?.show();
    setTimeout(() => this.poll(), 500);
  }

  setUp(up) {
    if (up === this.up) return;
    this.up = up;
    $('.dot', this.dot).classList.toggle('up', up);
    log(`${this.name}: ${up ? 'connected' : `not reachable (${this.error})`}`, !up);
  }

  stop() { return STOPS[this.kind](this); }
}

class Card {
  constructor(robot, picture) {
    this.robot = robot;
    this.el = $('#card').content.firstElementChild.cloneNode(true);
    this.el.style.setProperty('--robot', robot.color);
    $('.name', this.el).textContent = robot.name;
    $('.open', this.el).href = `${robot.url}/`;
    $('.stop', this.el).onclick = () => this.stopTimed();
    $('.popout', this.el).onclick = () =>
      window.open(`/?solo=${robot.id}`, `robot-${robot.id}`, 'width=760,height=940');
    this.img = $('.view img', this.el);
    this.img.alt = `${robot.name}: ${picture}`;
    $('.nosignal', this.el).textContent = `no picture from ${robot.name} yet`;
    robot.card = this;
  }

  async stopTimed() {
    const t0 = performance.now();
    this.release?.();
    try {
      await this.robot.stop();
      log(`${this.robot.name}: stopped (${Math.round(performance.now() - t0)} ms)`);
    } catch (e) {
      log(`${this.robot.name}: STOP failed -- ${e.message}`, true);
    }
  }

  act(label, promise) {
    promise.then(() => log(`${this.robot.name}: ${label}`))
      .catch(e => log(`${this.robot.name}: ${label} refused -- ${e.message}`, true));
  }

  offline() {   // before the first answer, or when the robot stops answering
    const first = this.robot.up === null;
    this.setPill(first ? 'connecting' : 'offline');
    $('.facts', this.el).textContent = first ? 'connecting...' : `not reachable: ${this.robot.error}`;
  }

  setPill(text, moving) {
    const pill = $('.mode', this.el);
    pill.textContent = text;
    pill.className = `pill mode${moving ? ' moving' : ''}${this.robot.up === false ? ' offline' : ''}`;
  }

  close() {
    this.closed = true;
    this.release?.();
    this.img.removeAttribute('src');   // ends a camera stream
    this.robot.card = null;
  }
}

class M3Card extends Card {
  constructor(robot) {
    super(robot, 'the arena map');
    const controls = $('#m3-controls').content.firstElementChild.cloneNode(true);
    $('.controls', this.el).append(controls);
    this.held = new Set();
    this.repeat = null;
    this.speed = $('input', controls);
    this.speed.oninput = () => { $('output', controls).textContent = `${this.speed.value} %`; };
    controls.querySelectorAll('[data-keys]').forEach(button => {
      const keys = button.dataset.keys.split(' ');
      button.addEventListener('pointerdown', e => {
        button.setPointerCapture(e.pointerId);
        keys.forEach(k => this.held.add(k));
        this.refresh();
      });
      const up = () => {
        if (keys.some(k => this.held.has(k))) { keys.forEach(k => this.held.delete(k)); this.refresh(); }
      };
      ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(t => button.addEventListener(t, up));
    });
    $('.halt', controls).onclick = () => this.stopTimed();
    this.autoButton = $('.auto', controls);
    this.autoButton.onclick = () => {
      const on = !this.robot.state?.auto;
      const stepwise = on && $('.stepwise input', controls).checked;
      this.act(on ? `auto loop started${stepwise ? ', stop and go' : ''}` : 'auto loop stopped',
        call(this.robot, 'api/auto', stepwise ? { on, step: 0.5 } : { on }));
    };
    this.goButton = $('.go', controls);
    this.goButton.onclick = () =>
      this.act('auto loop goes on', call(this.robot, 'api/auto', { resume: true }));
    this.img.onload = () => { this.img.classList.remove('broken'); this.nextMap(800); };
    this.img.onerror = () => { this.img.classList.add('broken'); this.nextMap(2000); };
    this.nextMap(0);
  }

  nextMap(delay) {
    setTimeout(() => {
      if (!this.closed) this.img.src = `${this.robot.url}/api/map.png?t=${Date.now()}`;
    }, delay);
  }

  key(k, down) {
    if (down) this.held.add(k); else this.held.delete(k);
    this.refresh();
  }

  refresh() {
    $('.controls', this.el).querySelectorAll('[data-keys]').forEach(b =>
      b.classList.toggle('on', b.dataset.keys.split(' ').every(k => this.held.has(k))));
    if (this.held.size) {   // the robot stops by itself 0.4 s after the last repeat
      this.send();
      if (!this.repeat) this.repeat = setInterval(() => this.send(), 100);
    } else if (this.repeat) {
      clearInterval(this.repeat);
      this.repeat = null;
      call(this.robot, 'api/stop', {}).catch(() => {});
    }
  }

  release() {   // let go of everything without a stop of its own (STOP sends one)
    this.held.clear();
    if (this.repeat) { clearInterval(this.repeat); this.repeat = null; }
    $('.controls', this.el).querySelectorAll('[data-keys]').forEach(b => b.classList.remove('on'));
  }

  send() {
    // One drive request at a time: if the robot answers slowly, repeats are skipped rather
    // than queued (a queue would take the browser's few connections per host from the rest).
    if (this.inflight) return;
    const v = [0, 0, 0];
    for (const k of this.held) DIRS[k].forEach((x, i) => { v[i] += x; });
    const scale = this.speed.value / 100;
    const [vx, vy, wz] = v.map(x => Math.max(-1, Math.min(1, x)) * scale);
    this.inflight = true;
    call(this.robot, 'api/drive', { vx, vy, wz }).then(() => { this.driveError = null; }).catch(e => {
      if (e.message !== this.driveError) log(`${this.robot.name}: drive refused -- ${e.message}`, true);
      this.driveError = e.message;
    }).finally(() => { this.inflight = false; });
  }

  show() {
    const s = this.robot.state;
    if (!s) {
      this.offline();
      return;
    }
    this.setPill(s.motion === false ? 'bench' : s.mode, s.mode !== 'idle' && s.mode !== 'paused');
    $('.mode', this.el).classList.toggle('paused', s.mode === 'paused');
    this.goButton.hidden = !s.paused;
    $('.note', this.el).textContent = s.note || '';
    $('.controls', this.el).querySelectorAll('button').forEach(b => {
      b.disabled = s.motion === false && !b.classList.contains('halt');   // bench: nothing moves
    });
    this.autoButton.classList.toggle('on', s.auto !== null);
    this.autoButton.textContent = s.auto
      ? `Auto: point ${s.auto[0] + 1} of ${s.auto[1]} (click to stop)` : 'Auto: loop the arena';
    const f = s.fence;
    $('.facts', this.el).textContent = [
      `pose     ${s.pose ? `x ${s.pose[0].toFixed(2)}  y ${s.pose[1].toFixed(2)}  ${s.pose[2].toFixed(0)} deg` : 'none yet'}`,
      `command  ${s.cmd.map(v => v.toFixed(2)).join('  ')}  (vx vy wz)`,
      `ahead    ${s.nearest_ahead_m === null ? 'clear' : `${s.nearest_ahead_m} m`}` +
        `     lidar ${s.scan_age_s !== null && s.scan_age_s < 0.5 ? 'ok' : 'STALE'}` +
        `     battery ${s.battery_v ?? '-'} V`,
      `fence    ${f ? `${f.size_m[0].toFixed(3)} x ${f.size_m[1].toFixed(3)} m${f.complete ? '' : ' (a wall not seen yet)'}, ${f.blobs} landmarks` : 'not fitted yet'}`,
    ].join('\n');
  }
}

class X3Card extends Card {
  constructor(robot) {
    super(robot, 'the wrist camera');
    const controls = $('#x3-controls').content.firstElementChild.cloneNode(true);
    $('.controls', this.el).append(controls);
    controls.querySelectorAll('[data-job]').forEach(b => {
      b.onclick = () => this.act(`${b.textContent}: started`, call(robot, 'api/auto/oneclick',
        { detector: 'auto', job: b.dataset.job }));
    });
    controls.querySelectorAll('[data-preset]').forEach(b => {
      b.onclick = () => this.act(`${b.textContent}: done`, call(robot, 'api/arm/preset', { name: b.dataset.preset }));
    });
    this.img.onload = () => this.img.classList.remove('broken');
    this.img.onerror = () => {   // the stream broke or never came: try again shortly
      this.img.classList.add('broken');
      setTimeout(() => this.stream(), 3000);
    };
    this.stream();
  }

  stream() {
    if (!this.closed) this.img.src = `${this.robot.url}/stream.mjpg?t=${Date.now()}`;
  }

  show() {
    const s = this.robot.state;
    if (!s) {
      this.offline();
      return;
    }
    const job = s.job;
    const running = job && job.status === 'running';
    this.setPill(running ? job.name : s.mode, running || s.arm.busy);
    $('.note', this.el).textContent = job ? `${job.name}: ${job.status}` : '';
    $('.facts', this.el).textContent = [
      `mode     ${s.mode}     arm ${s.arm.connected ? (s.arm.busy ? 'moving' : 'ready') : `not connected (${s.arm.error})`}` +
        `     battery ${s.arm.battery ?? '-'} V`,
      `camera   ${s.camera_error ? s.camera_error : `${s.camera_fps} fps`}     view ${s.view}, detector ${s.detector}`,
      `job      ${job ? `${job.name} ${job.status} ${job.elapsed_s} s${job.message ? ` -- ${job.message}` : ''}` : 'none'}`,
    ].join('\n');
  }
}

const CARDS = { m3: M3Card, x3: X3Card };

async function main() {
  const robots = (await (await fetch('/robots.json')).json()).map(spec => new Robot(spec));
  const solo = new URLSearchParams(location.search).get('solo');
  let cards = [];

  function show(layout) {
    cards.forEach(c => c.close());
    const visible = layout === 'both' ? robots : robots.filter(r => r.id === layout);
    cards = visible.map(r => new CARDS[r.kind](r));
    $('#cards').replaceChildren(...cards.map(c => c.el));
    $('#cards').style.setProperty('--cols', cards.length);
    cards.forEach(c => c.show());
    document.querySelectorAll('#layouts button').forEach(b => b.classList.toggle('on', b.dataset.layout === layout));
    if (!solo) store.set('console.layout', layout);
    $('#hint').hidden = !cards.some(c => c instanceof M3Card);
    document.title = layout === 'both' ? 'RoboARM console' : `${visible[0].name} -- RoboARM console`;
  }

  for (const [label, layout] of [...robots.map(r => [r.name, r.id]), ['Both', 'both']]) {
    const b = document.createElement('button');
    b.textContent = label;
    b.dataset.layout = layout;
    b.onclick = () => show(layout);
    $('#layouts').append(b);
  }
  if (solo) document.body.classList.add('solo');
  const saved = store.get('console.layout', 'both');
  show(solo || (saved === 'both' || robots.some(r => r.id === saved) ? saved : 'both'));

  async function stopAll() {
    const t0 = performance.now();
    cards.forEach(c => c.release?.());
    const done = await Promise.allSettled(robots.map(r => r.stop().then(() => performance.now() - t0)));
    const failed = done.some(d => d.status === 'rejected');
    log('STOP ALL: ' + robots.map((r, i) => done[i].status === 'fulfilled'
      ? `${r.name} ${Math.round(done[i].value)} ms` : `${r.name} FAILED (${done[i].reason.message})`).join(', '), failed);
  }
  $('#stopall').onclick = stopAll;

  const m3 = () => cards.find(c => c instanceof M3Card);
  addEventListener('keydown', e => {
    const typing = e.target instanceof Element && e.target.matches('input[type=text], textarea');
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === ' ' || e.key === 'Escape') { e.preventDefault(); stopAll(); return; }
    const k = e.key.toLowerCase();
    if (DIRS[k] && m3()) { e.preventDefault(); if (!e.repeat) m3().key(k, true); }
  });
  addEventListener('keyup', e => { const k = e.key.toLowerCase(); if (DIRS[k] && m3()) m3().key(k, false); });
  // A page in the background has its timers slowed to about once a second, so a held key
  // would go on nudging the robot: let go when the window loses focus or is hidden.
  addEventListener('blur', () => m3()?.release());
  document.addEventListener('visibilitychange', () => { if (document.hidden) m3()?.release(); });
}

main();
