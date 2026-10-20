'use strict';
// The event page (2026-10-22, out of the lab): each arm in robots.json side by side -- its camera
// live, what it is doing in one sentence, big buttons for us; H hides the buttons (what visitors
// see), SK / EN. The console (index.html, console.js) stays for everything else. JSON goes through
// serve.py as /robot/<id>/<path>; pictures come straight from the robot. voice.js, loaded after
// this, uses $, log and window.consoleActions from here.

const $ = (sel, root = document) => root.querySelector(sel);
const sleep = ms => new Promise(done => setTimeout(done, ms));
const ARMS = ['m3arm', 'x3arm'];   // the arms shown, in this order
const STOPS = {   // what STOP means for each kind of robot (as on the console)
  m3: ['api/stop'], x3: ['api/auto/stop', 'api/arm/hold'], m3arm: ['pick/stop'], x3arm: ['pick/stop'],
};
const LOW_V = { m3arm: 10.8, x3arm: 10.5 };   // battery warnings, as on the console
const store = {   // per-viewer conveniences only; the page works without them
  get(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ } },
};

const TEXT = {
  sk: {
    visitorView: 'Pre návštevníkov', controls: 'Ovládanie', other: 'EN', talk: 'Hovor (drž V)', listening: 'počúvam',
    heard: 'Počul som:', typed: 'alebo napíš: uprac stôl, X3 uprac stôl, domov, stop',
    keys: 'H skryje tlačidlá · medzerník zastaví všetko',
    cleanup: 'Uprac stôl', look: 'Pozri sa', home: 'Domov', teach: 'Nauč sa', find: 'Nájdi', pickNamed: 'Zober',
    thing: 'názov veci, napr. marhuľa', pretend: '(simulácia)', noArms: 'V robots.json nie je žiadna ruka.',
    connecting: 'Pripájam sa…', offline: 'Nepripojený', nosignal: 'Zatiaľ bez obrazu', cameraStalled: 'Kamera stojí',
    pillBusy: 'pracuje', pillIdle: 'čaká', pillOffline: 'nepripojený',
    ready: 'Pripravený', done: 'Hotovo', stopped: 'Zastavený', nothing: 'Nič v dosahu na zobratie',
    failed: 'Niečo sa nepodarilo',
    starting: 'Začínam', engage: 'Pripravujem ruku', lookBox: 'Hľadám krabicu', lookWide: 'Pozerám sa po stole',
    lookClose: 'Pozerám sa zblízka', lookAfter: 'Kontrolujem, či sa podarilo', taking: 'Beriem kúsok',
    takingThing: 'Beriem {thing}', drop: 'Nesiem ho do krabice', dropLine: 'Odkladám ho vpravo',
    putBack: 'Vraciam ho späť', goHome: 'Idem domov', drive: 'Podchádzam bližšie', driveBack: 'Vraciam sa na štart',
    lookTeach: 'Učím sa novú vec', teachIdentify: 'Spoznávam ju', lookFind: 'Hľadám veci, ktoré poznám',
    lookScan: 'Skenujem',
    picked: 'Zobral som {thing} za {s} s', missed: 'Nevyšlo to, skúsim znova',
    leftMissed: 'Jeden kúsok nechávam: dvakrát som ho minul', leftRefused: 'Jeden kúsok nechávam: tam nedočiahnem',
    clear: 'Stôl je uprataný', clearLeft: 'Upratané, {n} som musel nechať', boxFound: 'Krabicu vidím',
    boxMoved: 'Krabica sa pohla, mierim na jej nové miesto', noBox: 'Krabicu nevidím, kladiem vpravo',
    dropped: 'Je v krabici', droppedLine: 'Položený vpravo',
    pickedN: 'Zobraté {n}', missedN: 'Minuté {n}', perPiece: '{s} s na kúsok', inView: 'Na stole vidím {n}',
    busy: 'Robot práve pracuje', needName: 'Najprv napíš názov veci', refused: 'Nešlo to: {why}',
    allStopped: 'Všetko zastavené', noRobot: '{name} nie je pripojený', anyColour: 'farbu zatiaľ nerozlišujem',
    pickAny: 'Zober kúsok', pickCube: 'Zober kocku', pickBlock: 'Zober kváder', pickThing: 'Zober: {name}',
    cleanupCubes: 'Uprac kocky', cleanupBlocks: 'Uprac kvádre', sortLater: 'Triediť roboty zatiaľ nevedia',
    cube: '{n} mm kocku', blockLying: 'ležiaci kváder', blockStanding: 'stojaci kváder',
  },
  en: {
    visitorView: 'Visitor view', controls: 'Controls', other: 'SK', talk: 'Talk (hold V)', listening: 'listening',
    heard: 'Heard:', typed: 'or type: clean up, X3 clear the table, go home, stop',
    keys: 'H hides the buttons · Space stops all',
    cleanup: 'Clean up', look: 'Look', home: 'Home', teach: 'Teach', find: 'Find', pickNamed: 'Pick it',
    thing: 'its name, e.g. apricot', pretend: '(pretend)', noArms: 'No arm in robots.json.',
    connecting: 'Connecting…', offline: 'Not connected', nosignal: 'No picture yet', cameraStalled: 'Camera stalled',
    pillBusy: 'working', pillIdle: 'idle', pillOffline: 'offline',
    ready: 'Ready', done: 'Done', stopped: 'Stopped', nothing: 'Nothing in reach to pick',
    failed: 'Something went wrong',
    starting: 'Starting', engage: 'Getting the arm ready', lookBox: 'Looking for the box', lookWide: 'Looking over the table',
    lookClose: 'Taking a close look', lookAfter: 'Checking it worked', taking: 'Picking it up',
    takingThing: 'Picking up {thing}', drop: 'Taking it to the box', dropLine: 'Putting it down on the right',
    putBack: 'Putting it back', goHome: 'Going home', drive: 'Driving closer', driveBack: 'Driving back to the start',
    lookTeach: 'Learning a new thing', teachIdentify: 'Telling what it is', lookFind: 'Looking for things I know',
    lookScan: 'Scanning',
    picked: 'Picked {thing} in {s} s', missed: 'Missed it, trying again',
    leftMissed: 'Leaving one: missed it twice', leftRefused: 'Leaving one: I cannot reach it there',
    clear: 'The table is clear', clearLeft: 'Cleaned up, {n} had to stay', boxFound: 'I see the box',
    boxMoved: 'The box moved: aiming at its new place', noBox: 'No box in sight: putting them on the right',
    dropped: 'In the box', droppedLine: 'Put down on the right',
    pickedN: 'Picked {n}', missedN: 'Missed {n}', perPiece: '{s} s a piece', inView: '{n} on the table',
    busy: 'Busy with another job', needName: "Type the thing's name first", refused: 'Not done: {why}',
    allStopped: 'Everything stopped', noRobot: '{name} is not connected', anyColour: 'colours not told apart yet',
    pickAny: 'Pick one', pickCube: 'Pick a cube', pickBlock: 'Pick a block', pickThing: 'Pick: {name}',
    cleanupCubes: 'Clean up the cubes', cleanupBlocks: 'Clean up the blocks', sortLater: 'The robots cannot sort yet',
    cube: 'the {n} mm cube', blockLying: 'the lying block', blockStanding: 'the standing block',
  },
};
let lang = store.get('demo.lang', 'sk') === 'en' ? 'en' : 'sk';
const t = (key, vars = {}) => (TEXT[lang][key] ?? key).replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? '');

const STEPS = {   // the job's step (shared/roboshared/task.py, the robots' bodies) -> the sentence
  starting: 'starting', engage: 'engage', 'look.box': 'lookBox', 'look.wide': 'lookWide', 'look.close': 'lookClose',
  'look.after': 'lookAfter', pregrasp: 'lookClose', descend: 'taking', close: 'taking', grip: 'taking', lift: 'taking',
  drop: 'drop', 'put back': 'putBack', home: 'goHome', drive: 'drive', 'drive back': 'driveBack',
  'look.teach': 'lookTeach', 'teach.embed': 'lookTeach', 'teach.identify': 'teachIdentify', 'look.find': 'lookFind',
  'look.scan': 'lookScan',
};
const WIDE = new Set(['look.wide', 'look.after']);   // the looks over the whole table: how many pieces are on it
const NEWS = [   // a line of the robot's log -> news for the visitors: [pattern, key, bad]
  [/^picked the (.+?): (\d+) s/, 'picked'],
  [/^NOT picked/, 'missed', true],
  [/^left the one .*missed twice/, 'leftMissed', true],
  [/^left the one/, 'leftRefused', true],
  [/^clean-up done.*\((\d+) left/, 'clearLeft'],
  [/^clean-up done/, 'clear'],
  [/^the box has moved/, 'boxMoved'],
  [/^no drop-off box/, 'noBox', true],
  [/^drop-off box at/, 'boxFound'],
  [/^dropped into the box/, 'dropped'],
  [/^dropped .* out on the right/, 'droppedLine'],
];

function thing(raw) {   // a kit piece's name in the log ("30 mm cube", "block lying") in the page's language
  const cube = raw.match(/^(\d+) mm cube$/);
  if (cube) return t('cube', { n: cube[1] });
  return { 'block lying': t('blockLying'), 'block standing': t('blockStanding') }[raw] ?? raw;
}

function log(text, error = false) {   // the browser's console: voice.js reports here too
  (error ? console.warn : console.log)(`${new Date().toLocaleTimeString()}  ${text}`);
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

function why(e) {   // a refusal, for the page
  return /^busy/.test(e.message) ? t('busy') : t('refused', { why: e.message });
}

class Arm {   // an arm's pick service (/pick/* on the M3's bridge or the X3's), a column of the page
  constructor(spec) {
    Object.assign(this, spec);
    this.short = spec.name.replace(/ arm\b/, '');
    this.state = null;
    this.up = null;
    this.camera = null;   // the bridge's /health camera
    this.since = null;    // the log read up to (null: not read yet -- what is there is the past, not news)
    this.thing = null;    // what it is picking up, from its log
    this.news = null;     // {key, vars, bad}
    this.detail = '';     // the log's last line, for us
    this.base = null;     // picked and missed when the job began: the job's own counts
    this.busy = false;
    this.step = null;
    this.inView = null;   // the pieces the last look over the table found (a close look sees a few only)
    this.looks = null;
    this.el = $('#robot').content.firstElementChild.cloneNode(true);
    this.el.style.setProperty('--robot', spec.color);
    this.pill = document.createElement('span');
    this.pill.className = 'pill';
    $('#pills').append(this.pill);
    this.live = $('.live', this.el);
    this.look = $('.look', this.el);
    this.look.onload = () => { this.look.hidden = false; };
    this.look.onclick = () => $('.view', this.el).classList.toggle('swapped');
    $('.cleanup', this.el).onclick = () => this.act('pick/cleanup', {});
    $('.lookover', this.el).onclick = () => this.act('pick/look', {});
    $('.home', this.el).onclick = () => this.act('pick/home', {});
    if (spec.kind === 'm3arm') this.teachRow($('.teach', this.el));
    else $('.teach', this.el).classList.add('none');
    this.stream();
    this.poll();
    this.follow();
    this.health();
  }

  teachRow(row) {   // teach a thing by showing it, find it, pick it by name (the M3: m3pro/roboarm/m3teach.py)
    const name = $('.thing', row);
    const known = document.createElement('datalist');   // what it knows already, offered as you type
    known.id = `${this.id}-taught`;
    name.setAttribute('list', known.id);
    row.append(known);
    name.onfocus = async () => {
      try {
        const { names } = await call(this, 'pick/taught');
        known.replaceChildren(...names.map(n => Object.assign(document.createElement('option'), { value: n })));
      } catch { /* offline: no suggestions */ }
    };
    const named = () => {
      const value = name.value.trim();
      if (!value) {
        this.refuse(t('needName'));
        name.focus();
      }
      return value;
    };
    $('.teach-it', row).onclick = () => { const n = named(); if (n) this.act('pick/teach', { name: n }); };
    $('.find', row).onclick = () => this.act('pick/find', {});
    $('.pick-named', row).onclick = () => { const n = named(); if (n) this.act('pick/named', { name: n }); };
  }

  async act(path, body) {
    try {
      await call(this, path, body);
      log(`${this.short}: ${path} started`);
    } catch (e) {
      this.refuse(why(e));
      log(`${this.short}: ${path} refused -- ${e.message}`, true);
    }
  }

  label() { return this.short.replace('(pretend)', t('pretend')); }

  refuse(text) {   // for a few seconds, where the log's last line goes
    this.refusal = { text, until: Date.now() + 5000 };
    this.show();
  }

  stream() {   // the camera, live: if the stream breaks or never comes, again shortly
    this.live.onload = () => this.live.classList.remove('broken');
    this.live.onerror = () => {
      this.live.classList.add('broken');
      setTimeout(() => this.stream(), 3000);
    };
    this.live.src = `${this.url}/stream.mjpg?t=${Date.now()}`;
  }

  async poll() {
    for (;;) {
      try {
        this.state = await call(this, 'pick/state');
        if (this.up === false) this.since = null;   // back: perhaps restarted, its log counting from 1 again
        this.up = true;
      } catch (e) {
        this.state = null;
        this.up = false;
        this.error = e.message;
      }
      this.show();
      await sleep(500);
    }
  }

  async follow() {   // the robot's narrated log: what it picks, the news
    for (;;) {
      try {
        const { lines } = await call(this, `pick/log?since=${this.since ?? 0}`);
        if (this.since === null) {
          this.since = lines.length ? lines[lines.length - 1].n : 0;
          this.detail = lines.length ? lines[lines.length - 1].text : '';
        } else {
          for (const line of lines) {
            this.since = line.n;
            this.read(line.text);
          }
        }
      } catch { /* offline: the state says so */ }
      await sleep(800);
    }
  }

  read(text) {
    this.detail = text;
    const target = text.match(/^the (\d+ mm cube|block lying|block standing):/);   // m3pick's grip line
    if (target) this.thing = target[1];
    for (const [pattern, key, bad] of NEWS) {
      const m = text.match(pattern);
      if (!m) continue;
      this.news = { key, bad: Boolean(bad), vars: key === 'picked' ? { raw: m[1], s: m[2] } : { n: m[1] } };
      break;
    }
    if (/^(picked the|NOT picked)/.test(text)) this.thing = null;
  }

  async health() {
    for (;;) {
      try {
        this.camera = (await call(this, 'health')).camera;
      } catch {
        this.camera = null;
      }
      await sleep(2000);
    }
  }

  sentence(s) {
    if (s.busy) {
      const key = STEPS[s.step];
      if (!key) return s.step;
      if (key === 'taking' && this.thing) return t('takingThing', { thing: thing(this.thing) });
      if (key === 'drop' && !s.box) return t('dropLine');
      return t(key);
    }
    if (s.error) {
      if (/^stopped/.test(s.error)) return t('stopped');
      if (/^[A-Z]\w*(Error|Exception)\b/.test(s.error)) return t('failed');
      return t('nothing');   // the job's NothingToPick: why not, in its words, below
    }
    return this.base ? t('done') : t('ready');
  }

  show() {
    const s = this.state;
    $('.name', this.el).textContent = this.label();
    $('.nosignal', this.el).textContent = t('nosignal');
    $('.camerr', this.el).textContent = this.camera?.error ? t('cameraStalled') : '';
    const doing = $('.doing', this.el), news = $('.news', this.el), counts = $('.counts', this.el);
    const detail = $('.detail', this.el), volts = $('.volts', this.el);
    this.pill.textContent = `${this.short.replace(' (pretend)', '')}: ${t(!s ? 'pillOffline' : s.busy ? 'pillBusy' : 'pillIdle')}`;
    this.pill.className = `pill${!s && this.up === false ? ' offline' : s?.busy ? ' busy' : ''}`;
    if (!s) {
      doing.textContent = t(this.up === false ? 'offline' : 'connecting');
      doing.classList.toggle('bad', this.up === false);
      news.textContent = counts.textContent = volts.textContent = '';
      detail.textContent = this.up === false ? this.error : '';
      return;
    }
    if (s.busy && !this.busy) {   // a job began: its own counts, no news from before
      this.base = { picked: s.picked, missed: s.missed };
      this.news = null;
      this.thing = null;
      this.inView = null;
    }
    this.busy = s.busy;
    if (WIDE.has(this.step) && s.step !== this.step) this.inView = s.found.filter(o => o.kind).length;   // as it ended
    this.step = s.step;
    doing.textContent = this.sentence(s);
    doing.classList.toggle('bad', !s.busy && Boolean(s.error) && !/^stopped/.test(s.error));
    const n = this.news;
    news.textContent = n ? t(n.key, n.key === 'picked' ? { thing: thing(n.vars.raw), s: n.vars.s } : n.vars) : '';
    news.classList.toggle('bad', Boolean(n?.bad));
    const parts = [];
    if (this.base) {
      parts.push(t('pickedN', { n: s.picked - this.base.picked }));
      if (s.missed > this.base.missed) parts.push(t('missedN', { n: s.missed - this.base.missed }));
      if (s.last?.seconds && s.picked > this.base.picked) parts.push(t('perPiece', { s: Math.round(s.last.seconds) }));
    }
    if (s.busy && this.inView) parts.push(t('inView', { n: this.inView }));
    counts.textContent = parts.join(' · ');
    const low = s.battery_v != null && s.battery_v < LOW_V[this.kind];
    volts.textContent = s.battery_v == null ? '' : `${s.battery_v.toLocaleString(lang === 'sk' ? 'sk-SK' : 'en-GB',
      { minimumFractionDigits: 1, maximumFractionDigits: 1 })} V`;
    volts.classList.toggle('low', low);
    const refusal = this.refusal && this.refusal.until > Date.now() ? this.refusal.text : '';
    detail.textContent = refusal || (s.busy ? `${s.job} · ${s.step} · ${Math.round(s.now - s.since)} s · ${this.detail}`
      : s.error || this.detail);
    detail.classList.toggle('bad', Boolean(refusal) || (!s.busy && Boolean(s.error)));
    if (s.looks !== this.looks) {
      this.looks = s.looks;
      if (s.looks) this.look.src = `${this.url}/pick/frame.jpg?n=${s.looks}`;
    }
  }
}

async function main() {
  const robots = await (await fetch('/robots.json')).json();
  const arms = ARMS.flatMap(kind => robots.filter(r => r.kind === kind)).map(spec => new Arm(spec));
  const list = $('#robots');
  if (arms.length) list.replaceChildren(...arms.map(a => a.el));
  else list.replaceChildren(Object.assign(document.createElement('p'), { className: 'empty', textContent: t('noArms') }));
  list.style.setProperty('--cols', Math.max(1, arms.length));

  function setLang(next) {
    lang = next;
    store.set('demo.lang', lang);
    document.documentElement.lang = lang;
    document.querySelectorAll('[data-t]').forEach(el => { el.textContent = t(el.dataset.t); });
    document.querySelectorAll('[data-t-placeholder]').forEach(el => { el.placeholder = t(el.dataset.tPlaceholder); });
    const talk = $('#talk');
    Object.assign(talk.dataset, { idle: t('talk'), listening: t('listening') });   // voice.js's words
    if (!talk.classList.contains('on')) talk.textContent = t('talk');
    $('#lang').textContent = t('other');
    setVisitor(document.body.classList.contains('visitor'));
    arms.forEach(a => a.show());
  }
  function setVisitor(on) {
    document.body.classList.toggle('visitor', on);
    $('#visitor').textContent = t(on ? 'controls' : 'visitorView');
    store.set('demo.visitor', on);
  }
  $('#lang').onclick = () => setLang(lang === 'sk' ? 'en' : 'sk');
  $('#visitor').onclick = () => setVisitor(!document.body.classList.contains('visitor'));
  setVisitor(store.get('demo.visitor', false));
  setLang(lang);

  async function stopAll() {
    const t0 = performance.now();
    const calls = robots.filter(r => STOPS[r.kind]).map(r =>
      Promise.all(STOPS[r.kind].map(path => call(r, path, {}))).then(() => `${r.name} ${Math.round(performance.now() - t0)} ms`));
    const done = await Promise.allSettled(calls);
    log(`STOP ALL: ${done.map(d => d.status === 'fulfilled' ? d.value : `FAILED (${d.reason.message})`).join(', ')}`,
      done.some(d => d.status === 'rejected'));
  }
  $('#stopall').onclick = stopAll;

  // What a voice command may do (voice.js calls these with the command voice.py found -- by the
  // language model, llm.py, or the grammar): each answers with what was done, or throws why not.
  function route(c) {   // [path, body, what it is called]
    const cleanup = { cube: 'cleanupCubes', block: 'cleanupBlocks' }[c.kind] ?? 'cleanup';
    return {
      pick: () => ['pick/start', { kind: c.kind ?? null, colour: c.colour, size_mm: c.size_mm },
        t({ cube: 'pickCube', block: 'pickBlock' }[c.kind] ?? 'pickAny')],
      clean_up: () => ['pick/cleanup', { kind: c.kind ?? null }, t(cleanup)],
      home: () => ['pick/home', {}, t('home')],
      look: () => ['pick/look', {}, t('look')],
      find: () => ['pick/find', {}, t('find')],
      pick_named: () => ['pick/named', { name: c.name }, t('pickThing', { name: c.name })],
      sort: () => { throw new Error(t('sortLater')); },   // a job the robots do not have yet
    }[c.action]();
  }
  async function one(kind, command) {
    const arm = arms.find(a => a.kind === kind);
    if (!arm || arm.up === false) throw new Error(t('noRobot', { name: kind === 'x3arm' ? 'X3 Plus' : 'M3 Pro' }));
    const [path, body, what] = route(command);
    try {
      await call(arm, path, body);
    } catch (e) {
      throw new Error(`${arm.label()}: ${why(e)}`);
    }
    const colour = command.action === 'pick' && (command.colour || command.size_mm);
    return `${arm.label()}: ${what}${colour ? ` (${t('anyColour')})` : ''}`;
  }
  async function job(command) {   // on the robot named, or on both
    const kinds = command.robot === 'both' ? ['m3arm', 'x3arm'] : [command.robot === 'x3' ? 'x3arm' : 'm3arm'];
    const done = await Promise.allSettled(kinds.map(kind => one(kind, command)));
    const said = done.map(d => d.status === 'fulfilled' ? d.value : d.reason.message).join(' · ');
    if (done.every(d => d.status === 'rejected')) throw new Error(said);
    return said;
  }
  window.consoleActions = {
    stop: async () => { await stopAll(); return t('allStopped'); },
    ...Object.fromEntries(['pick', 'clean_up', 'home', 'look', 'find', 'pick_named', 'sort'].map(a => [a, job])),
  };

  addEventListener('keydown', e => {
    if (e.key === 'Escape') { e.preventDefault(); stopAll(); return; }   // even while typing
    const typing = e.target instanceof Element && e.target.matches('input, textarea, select');
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === ' ') { e.preventDefault(); stopAll(); return; }
    if (e.key.toLowerCase() === 'h' && !e.repeat) setVisitor(!document.body.classList.contains('visitor'));
  });
}

main();
