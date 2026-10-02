'use strict';
// The console page: every robot in robots.json is polled all the time (so STOP ALL and the
// status dots cover robots whose card is hidden); a card per robot shown. JSON goes through
// serve.py as /robot/<id>/<path>; pictures come straight from the robot.

const $ = (sel, root = document) => root.querySelector(sel);
const DIRS = { w: [1, 0, 0], s: [-1, 0, 0], a: [0, 1, 0], d: [0, -1, 0], q: [0, 0, 1], e: [0, 0, -1] };
const STOPS = {   // what STOP means for each kind of robot
  m3: robot => call(robot, 'api/stop', {}),
  x3: robot => Promise.all([call(robot, 'api/auto/stop', {}), call(robot, 'api/arm/hold', {})]),
  m3arm: robot => call(robot, 'pick/stop', {}),   // the M3's arm: the pick service on its bridge
};
const STATES = { m3arm: 'pick/state' };
const LOW_V = 10.8;   // the M3's battery: its arm refuses below 10.5 V, the board went silent at ~10.4   // where a kind of robot says how it is, if not api/state
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
      this.state = await call(this, STATES[this.kind] ?? 'api/state');
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
    this.surveyButton = $('.survey', controls);
    this.surveyButton.onclick = () => {
      const stepwise = $('.stepwise input', controls).checked;
      this.act(`survey started${stepwise ? ', stop and go' : ''}`, call(this.robot, 'api/auto',
        stepwise ? { on: true, plan: 'survey', step: 0.5 } : { on: true, plan: 'survey' }));
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
      ? `Auto ${s.plan}: ${s.plan === 'survey' ? 'station' : 'point'} ${s.auto[0] + 1} of ${s.auto[1]} (click to stop)`
      : 'Auto: loop the arena';
    this.surveyButton.hidden = s.auto !== null;
    const f = s.fence;
    $('.facts', this.el).textContent = [
      `pose     ${s.pose ? `x ${s.pose[0].toFixed(2)}  y ${s.pose[1].toFixed(2)}  ${s.pose[2].toFixed(0)} deg` : 'none yet'}`,
      `command  ${s.cmd.map(v => v.toFixed(2)).join('  ')}  (vx vy wz)`,
      `ahead    ${s.nearest_ahead_m === null ? 'clear' : `${s.nearest_ahead_m} m`}` +
        `     lidar ${s.scan_age_s !== null && s.scan_age_s < 0.5 ? 'ok' : 'STALE'}` +
        `     battery ${s.battery_v ?? '-'} V`,
      `fence    ${f ? `${f.size_m[0].toFixed(3)} x ${f.size_m[1].toFixed(3)} m${f.complete ? '' : ' (a wall not seen yet)'}, ${f.blobs} landmarks` : 'not fitted yet'}`,
      ...(s.plan === 'survey' ? [`station  ${s.recording ? 'standing: the sharp frames' : 'on the way'}`] : []),
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

class M3ArmCard extends Card {   // the M3's arm and its pick, run on the robot (m3pro/roboarm/m3pick.py)
  constructor(robot) {
    super(robot, 'the arm camera, live');
    const controls = $('#m3arm-controls').content.firstElementChild.cloneNode(true);
    $('.controls', this.el).append(controls);
    const drive = () => $('.drive', controls).checked;
    controls.querySelectorAll('[data-pick]').forEach(b => {
      b.onclick = () => this.act(`${b.textContent}: started`,
        call(robot, 'pick/start', { kind: b.dataset.pick || null, drive: drive() }));
    });
    $('.cleanup', controls).onclick = () => this.act('clean up: started', call(robot, 'pick/cleanup', { drive: drive() }));
    $('.home', controls).onclick = () => this.act('home: started', call(robot, 'pick/home', {}));
    this.speed = $('.armspeed input', controls);
    this.speed.oninput = () => { $('.armspeed output', controls).textContent = `${this.speed.value} %`; };
    this.speed.onchange = () => this.setting('speed', this.speed.value);
    this.fields = $('.tuning .fields', controls);
    this.settingsLoaded = false;
    this.timing = $('.timing tbody', controls);
    this.total = $('.timing tfoot', controls);
    this.armlog = $('.armlog', controls);
    const volts = document.createElement('span');   // the battery, in the card's head
    volts.className = 'pill volts';
    volts.title = "the M3's battery (the arm refuses below 10.5 V)";
    $('.mode', this.el).after(volts);
    this.look = document.createElement('img');   // the last look, marked: what the robot made of it
    this.look.className = 'look';
    this.look.title = 'the last look, marked: click to swap with the live picture';
    this.look.alt = `${robot.name}: the last look`;
    this.look.onclick = () => $('.view', this.el).classList.toggle('swapped');
    $('.view', this.el).append(this.look);
    this.looks = null;
    this.since = 0;
    this.img.onload = () => this.img.classList.remove('broken');
    this.img.onerror = () => {   // the stream broke or never came: try again shortly
      this.img.classList.add('broken');
      setTimeout(() => this.stream(), 3000);
    };
    this.stream();
    this.follow();
  }

  stream() {
    if (!this.closed) this.img.src = `${this.robot.url}/stream.mjpg?t=${Date.now()}`;
  }

  showSettings(list) {   // the speed slider, and a field for each other speed and pause
    const rows = [];
    for (const s of list) {
      if (s.key === 'speed') {
        this.speed.value = s.value;
        this.speed.oninput();
        continue;
      }
      const label = document.createElement('label');
      label.textContent = s.label;
      label.title = `${s.min}-${s.max} ${s.unit}, ${s.default} at first`;
      const input = document.createElement('input');
      Object.assign(input, { type: 'number', min: s.min, max: s.max, step: s.step, value: s.value });
      input.dataset.key = s.key;
      input.onchange = () => this.setting(s.key, input.value, input);
      label.htmlFor = input.id = `m3arm-${s.key}`;
      const unit = document.createElement('span');
      unit.className = 'unit';
      unit.textContent = s.unit;
      rows.push(label, input, unit);
    }
    this.fields.replaceChildren(...rows);
    this.settingsLoaded = true;
  }

  async setting(key, value, input) {   // takes effect from the robot's next move
    try {
      const { settings } = await call(this.robot, 'pick/settings', { [key]: Number(value) });
      const now = settings.find(s => s.key === key);
      if (input) {
        input.value = now.value;   // held to its range by the robot
        input.classList.add('saved');
        setTimeout(() => input.classList.remove('saved'), 800);
      } else {
        this.speed.value = now.value;
        this.speed.oninput();
      }
      log(`${this.robot.name}: ${now.label} ${now.value} ${now.unit}`);
    } catch (e) {
      log(`${this.robot.name}: ${key} not changed -- ${e.message}`, true);
    }
  }

  async follow() {   // the robot's narrated steps, and where the time went, while the card is open
    let tick = 0;
    while (!this.closed) {
      try {
        if (!this.settingsLoaded) this.showSettings((await call(this.robot, 'pick/settings')).settings);
        const { lines } = await call(this.robot, `pick/log?since=${this.since}`);
        for (const line of lines) {
          this.since = line.n;
          const div = document.createElement('div');
          div.textContent = `${new Date(line.t * 1000).toLocaleTimeString()}  ${line.text}`;
          if (/NOT picked|stopped|failed|Error/.test(line.text)) div.className = 'err';
          this.armlog.prepend(div);
          while (this.armlog.childElementCount > 60) this.armlog.lastElementChild.remove();
        }
        if (tick++ % 3 === 0) this.showTiming(await call(this.robot, 'api/metrics?events=0'));
      } catch { /* offline: the facts say so */ }
      await new Promise(done => setTimeout(done, 700));
    }
  }

  showTiming(m) {
    const run = m.runs[0];
    if (!run) return;
    const rows = Object.entries(run.top).map(([name, t]) => {
      const all = m.stages[name];
      return `<tr><td>${name}</td><td>${t.total_s.toFixed(1)}</td><td>${all ? (all.mean_ms / 1000).toFixed(1) : '-'}</td>` +
        `<td>${all ? all.count : ''}</td></tr>`;
    });
    this.timing.innerHTML = rows.join('');
    const detect = m.stages.detect;
    this.total.innerHTML = `<tr><td>${run.name}: ${run.status}</td><td>${run.total_s.toFixed(1)}</td>` +
      `<td colspan="2">${detect ? `detect ${(detect.mean_ms / 1000).toFixed(2)} s a look` : ''}</td></tr>`;
  }

  show() {
    const s = this.robot.state;
    if (!s) {
      this.offline();
      return;
    }
    this.setPill(s.busy ? `${s.job.replace('_', ' ')}: ${s.step}` : 'idle', s.busy);
    $('.volts', this.el).textContent = s.battery_v == null ? '' : `${s.battery_v.toFixed(1)} V`;
    $('.volts', this.el).classList.toggle('low', s.battery_v != null && s.battery_v < LOW_V);
    $('.note', this.el).textContent = s.busy ? `${Math.round(s.now - s.since)} s in this step` : (s.error || '');
    if (s.looks !== this.looks) {
      this.looks = s.looks;
      if (s.looks) this.look.src = `${this.robot.url}/pick/frame.jpg?n=${s.looks}`;
    }
    const last = s.last;
    const seen = s.found.filter(o => o.kind)
      .map(o => `${o.name} ${(100 * o.x).toFixed(0)},${(100 * o.y).toFixed(0)} cm${o.reach ? '' : ' (out of reach)'}`);
    $('.facts', this.el).textContent = [
      `job      ${s.busy ? `${s.job.replace('_', ' ')} -- ${s.step}` : 'none'}     picked ${s.picked}, missed ${s.missed}`,
      `battery  ${s.battery_v == null ? '-' : `${s.battery_v.toFixed(2)} V`}` +
        `${s.battery_v != null && s.battery_v < LOW_V ? '   LOW: below ' + LOW_V + ' V the arm stops -- charge it' : ''}`,
      `last     ${last ? `${last.picked ? 'picked' : 'MISSED'} the ${last.kind} in ${last.seconds} s` +
        ` (approach ${last.pitch} deg, fingers along ${last.yaw_deg} deg from the ${last.yaw_from} look, J5 ${last.j5})` : '-'}`,
      `in view  ${seen.length ? seen.join('; ') : 'no cube or block'}`,
      `box      ${s.box ? `at ${(100 * s.box.x).toFixed(1)}, ${(100 * s.box.y).toFixed(1)} cm, turned ${s.box.turn.toFixed(0)} deg;` +
        ` ${s.box.in_it} in it when seen, ${s.box.drops} dropped in since` : 'not seen: drops go on the line out on the right'}`,
      `base     ${s.moved_cm && (s.moved_cm[0] || s.moved_cm[1])
        ? `${s.moved_cm[0]} cm forward, ${s.moved_cm[1]} cm left of where the job began` : 'where the job began'}`,
    ].join('\n');
  }
}

class CloudCard {   // the M3's 3D scans: not a robot -- nothing polled, no STOP of its own
  constructor() {
    this.el = $('#cloud-card').content.firstElementChild.cloneNode(true);
    this.scene = $('.scene', this.el);
    this.pick = $('.pick', this.el);
    this.print = $('.print', this.el);
    this.clouds = [];
    this.facts = { scan: [], floor: '' };
    this.viewer = import('./cloud.js').then(m => m.viewer($('.gl', this.el)));   // WebGL only when shown
    this.viewer.catch(e => this.fail(e));
    this.shown = [...this.el.querySelectorAll('[data-show]')];   // 3D, or the floor print
    this.shown.forEach(b => {
      b.onclick = () => {
        this.shown.forEach(c => c.classList.toggle('on', c === b));
        this.el.classList.toggle('floor', b.dataset.show === 'floor');
        store.set('console.cloud.show', b.dataset.show);
      };
    });
    this.shown.find(b => b.dataset.show === store.get('console.cloud.show', '3d'))?.click();
    $('.popout', this.el).onclick = () => window.open('/?solo=cloud', 'cloud', 'width=1100,height=860');
    $('.home', this.el).onclick = () => this.viewer.then(v => v.reset());
    const colours = [...this.el.querySelectorAll('[data-colour]')];
    colours.forEach(b => {
      b.onclick = () => {
        colours.forEach(c => c.classList.toggle('on', c === b));
        store.set('console.cloud.colour', b.dataset.colour);
        this.viewer.then(v => v.setColour(b.dataset.colour));
      };
    });
    colours.find(b => b.dataset.colour === store.get('console.cloud.colour', 'camera'))?.click();
    const size = $('.cloud-controls input', this.el);
    size.oninput = () => {
      $('.cloud-controls output', this.el).textContent = `${size.value} px`;
      store.set('console.cloud.size', Number(size.value));
      this.viewer.then(v => v.setSize(Number(size.value)));
    };
    size.value = store.get('console.cloud.size', 3);
    size.oninput();
    this.pick.onchange = () => this.fetchCloud(this.pick.value);
    $('.file input', this.el).onchange = e => { if (e.target.files[0]) this.openFile(e.target.files[0]); };
    this.scene.addEventListener('dragover', e => { e.preventDefault(); this.scene.classList.add('dropping'); });
    this.scene.addEventListener('dragleave', () => this.scene.classList.remove('dropping'));
    this.scene.addEventListener('drop', e => {
      e.preventDefault();
      this.scene.classList.remove('dropping');
      if (e.dataTransfer.files[0]) this.openFile(e.dataTransfer.files[0]);
    });
    this.list();
  }

  async list() {
    try {
      const clouds = this.clouds = await (await fetch('/clouds.json')).json();
      this.pick.replaceChildren(...clouds.map(c =>
        new Option(`${c.name}, ${c.written}, ${(c.bytes / 1e6).toFixed(1)} MB`, c.name)));
      if (!clouds.length) {
        this.message('no 3D scan on this laptop yet -- m3pro/tools/arena_map.py rtab-export, or open a .ply');
        return;
      }
      const saved = store.get('console.cloud', null);
      this.pick.value = clouds.some(c => c.name === saved) ? saved : clouds[0].name;
      this.fetchCloud(this.pick.value);
    } catch (e) {
      this.fail(e);
    }
  }

  async fetchCloud(name) {
    store.set('console.cloud', name);
    this.pick.querySelector('option[value=""]')?.remove();   // a file opened from this computer
    const token = this.token = {};   // a later choice wins over a slower earlier one
    this.setPrint(this.clouds.find(c => c.name === name)?.floor);
    this.message(`loading ${name}...`);
    try {
      const reply = await fetch(`/clouds/${encodeURIComponent(name)}`);
      if (!reply.ok) throw new Error(`${name}: HTTP ${reply.status}`);
      const buffer = await reply.arrayBuffer();
      if (token === this.token) await this.display(name, buffer);
    } catch (e) {
      if (token === this.token) this.fail(e);
    }
  }

  async openFile(file) {
    const token = this.token = {};
    // listed while it is shown, so that choosing an exported scan again is a change
    this.pick.querySelector('option[value=""]')?.remove();
    const option = new Option(`${file.name} (this computer)`, '', true, true);
    option.disabled = true;
    this.pick.prepend(option);
    this.setPrint(null);
    this.message(`reading ${file.name}...`);
    try {
      const buffer = await file.arrayBuffer();
      if (token === this.token) await this.display(`${file.name} (opened from this computer)`, buffer);
    } catch (e) {
      if (token === this.token) this.fail(e);
    }
  }

  async display(name, buffer) {
    const t0 = performance.now();
    const { count, size } = (await this.viewer).load(buffer);
    this.message('');
    $('.note', this.el).textContent = `${count.toLocaleString('en')} points`;
    this.facts.scan = [
      `scan     ${name}`,
      `points   ${count.toLocaleString('en')}, ${(buffer.byteLength / 1e6).toFixed(1)} MB, ` +
        `on screen in ${Math.round(performance.now() - t0)} ms`,
      `extent   ${size.map(v => v.toFixed(2)).join(' x ')} m  (x y z, 1st to 99th percentile)`,
    ];
    this.writeFacts();
    log(`3D scan: ${name}, ${count.toLocaleString('en')} points`);
  }

  setPrint(file) {
    // The floor print exported with the scan (1 cm a pixel, the map's x to the right), turned a
    // quarter to the left when it stands upright, so that it lies as the 3D view starts: the
    // long side across, the same way round.
    this.printFile = file;
    const floorButton = this.shown.find(b => b.dataset.show === 'floor');
    floorButton.disabled = !file;
    floorButton.title = file ? '' : 'no floor print with this scan';
    if (!file && this.el.classList.contains('floor')) this.shown[0].click();
    this.print.width = this.print.height = 0;
    this.facts.floor = `floor    ${file ? `${file}, loading` : 'no print with this scan'}`;
    this.writeFacts();
    if (!file) return;
    const image = new Image();
    image.onload = () => {
      if (file !== this.printFile) return;   // another scan chosen meanwhile
      const turn = image.height > image.width;
      this.print.width = turn ? image.height : image.width;
      this.print.height = turn ? image.width : image.height;
      const ctx = this.print.getContext('2d');
      if (turn) { ctx.translate(0, this.print.height); ctx.rotate(-Math.PI / 2); }
      ctx.drawImage(image, 0, 0);
      this.facts.floor = `floor    ${file}, ${(image.width / 100).toFixed(2)} x ` +
        `${(image.height / 100).toFixed(2)} m at 1 cm a pixel${turn ? ' (shown turned a quarter left)' : ''}`;
      this.writeFacts();
    };
    image.onerror = () => {
      if (file !== this.printFile) return;
      this.facts.floor = `floor    ${file} did not load`;
      this.writeFacts();
    };
    image.src = `/clouds/${encodeURIComponent(file)}`;
  }

  writeFacts() { $('.facts', this.el).textContent = [...this.facts.scan, this.facts.floor].join('\n'); }

  message(text) { $('.nosignal', this.el).textContent = text; }

  fail(e) {
    this.message(`could not show the scan: ${e.message}`);
    log(`3D scan: ${e.message}`, true);
  }

  show() {}

  close() {
    this.closed = true;
    this.viewer.then(v => v.dispose()).catch(() => {});
  }
}

const CARDS = { m3: M3Card, x3: X3Card, m3arm: M3ArmCard };

async function main() {
  const robots = (await (await fetch('/robots.json')).json()).map(spec => new Robot(spec));
  const carded = robots.filter(r => CARDS[r.kind]);
  const solo = new URLSearchParams(location.search).get('solo');
  let cards = [];

  function show(layout) {
    cards.forEach(c => c.close());
    const visible = layout === 'both' ? carded : carded.filter(r => r.id === layout);
    cards = layout === 'cloud' ? [new CloudCard()] : visible.map(r => new CARDS[r.kind](r));
    $('#cards').replaceChildren(...cards.map(c => c.el));
    $('#cards').style.setProperty('--cols', cards.length);
    cards.forEach(c => c.show());
    document.querySelectorAll('#layouts button').forEach(b => b.classList.toggle('on', b.dataset.layout === layout));
    if (!solo) store.set('console.layout', layout);
    $('#hint').hidden = !cards.some(c => c instanceof M3Card);
    document.title = layout === 'both' ? 'RoboARM console'
      : `${layout === 'cloud' ? '3D scan' : visible[0].name} -- RoboARM console`;
  }

  const layouts = [...carded.map(r => [r.name, r.id]), ['Both', 'both'], ['3D scan', 'cloud']];
  for (const [label, layout] of layouts) {
    const b = document.createElement('button');
    b.textContent = label;
    b.dataset.layout = layout;
    b.onclick = () => show(layout);
    $('#layouts').append(b);
  }
  if (solo) document.body.classList.add('solo');
  const saved = store.get('console.layout', 'both');
  show(solo || (layouts.some(([, layout]) => layout === saved) ? saved : 'both'));

  async function stopAll() {
    const t0 = performance.now();
    cards.forEach(c => c.release?.());
    const done = await Promise.allSettled(robots.map(r => r.stop().then(() => performance.now() - t0)));
    const failed = done.some(d => d.status === 'rejected');
    log('STOP ALL: ' + robots.map((r, i) => done[i].status === 'fulfilled'
      ? `${r.name} ${Math.round(done[i].value)} ms` : `${r.name} FAILED (${done[i].reason.message})`).join(', '), failed);
  }
  $('#stopall').onclick = stopAll;

  // What a voice command may do (voice.js calls these with the command it heard): each answers
  // with what was done, or throws why not. The X3's are its card's buttons. The M3's arm is a
  // robot of its own in robots.json -- id and kind "m3arm", the pick service on the M3's bridge
  // (http://192.168.73.66:8761) -- and until it is listed there it is "not connected".
  const JOBS = {   // [path, body, what it is called]
    x3: {
      pick: () => ['api/auto/oneclick', { detector: 'auto', job: 'pick' }, 'Pick one started'],
      clean_up: () => ['api/auto/oneclick', { detector: 'auto', job: 'pickall' }, 'Clear the table started'],
      home: () => ['api/arm/preset', { name: 'home' }, 'Home done'],
    },
    m3arm: {
      pick: c => ['pick/start', { kind: c.kind, colour: c.colour, size_mm: c.size_mm }, 'pick started'],
      clean_up: () => ['pick/cleanup', {}, 'clean up started'],
      home: () => ['pick/home', {}, 'home'],
    },
  };
  async function job(command) {
    const kind = command.robot === 'x3' ? 'x3' : 'm3arm';
    const robot = robots.find(r => r.id === kind);
    if (!robot) throw new Error(kind === 'x3' ? 'no X3 in robots.json' : 'M3 arm not connected');
    const [path, body, done] = JOBS[kind][command.action](command);
    await call(robot, path, body);
    const asked = kind === 'x3' && [command.colour, command.size_mm && `${command.size_mm} mm`].filter(Boolean);
    return `${robot.name}: ${done}${asked?.length ? ` (it takes what it finds first, not "${asked.join(' ')}")` : ''}`;
  }
  window.consoleActions = {
    stop: async () => { await stopAll(); return 'STOP ALL (the log has the times)'; },
    pick: job,
    clean_up: job,
    home: job,
  };

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
