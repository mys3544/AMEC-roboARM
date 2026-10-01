'use strict';
// Voice commands: hold Talk (or V), say a command, let go. The page records the microphone and
// sends what was said, as a 16 kHz mono 16-bit WAV, to /api/voice (serve.py -> voice.py:
// Parakeet on this laptop, then a small grammar), and carries out the command it gets back
// through window.consoleActions (console.js): stop, pick, clean_up, home. A typed command takes
// the same path without the recogniser. Loaded after console.js, whose $ and log it uses; not in
// a popped-out window (one microphone, one page that listens).

(() => {
  if (new URLSearchParams(location.search).has('solo')) return;
  const RATE = 16000;      // what the recogniser wants
  const KEY = 'v';         // hold to talk (W/A/S/D/Q/E drive, Space and Esc stop)
  const PREROLL_S = 0.3;   // kept from before the press: a word begun a moment early is not cut
  const SHORTEST_S = 0.25; // a tap is not a command
  const LONGEST_S = 15;    // a key held longer is sent as it is
  const KEPT = 12;         // commands in the history
  // The microphone's samples, as they come (128 at a time), from a worklet: no library, no file.
  const TAP = `registerProcessor('tap', class extends AudioWorkletProcessor {
    process([input]) { if (input[0]) this.port.postMessage(input[0].slice()); return true; }
  });`;

  const box = $('#voice');
  const button = $('#talk');
  let mic = null;          // the open microphone, { context }, from the first press on
  let rate = 48000;        // its sample rate
  let chunks = [];         // what it heard since the press, with the pre-roll before it
  let length = 0;          // samples in chunks
  let held = null;         // when the press began (performance.now()), while held
  let peak = 0;            // the loudest sample since the last tick: the level on the button
  let ticker = null;

  function setState(text, error = false) {
    const state = $('.state', box);
    state.textContent = text;
    state.classList.toggle('err', error);
  }

  async function status() {   // also starts the model's load in serve.py
    try {
      const s = await (await fetch('/api/voice')).json();
      if (s.ready) setState(`ready -- Parakeet TDT 0.6B v3 on this laptop, loaded in ${s.load_s} s`);
      else if (s.loading) { setState('loading the speech model...'); setTimeout(status, 1000); }
      else setState(`no speech recognition: ${s.error} -- ${s.fix}. Typed commands work.`, true);
    } catch (e) {
      setState(`voice: ${e.message}`, true);
    }
  }

  async function openMic() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    const context = new AudioContext();
    rate = context.sampleRate;
    await context.audioWorklet.addModule(URL.createObjectURL(new Blob([TAP], { type: 'text/javascript' })));
    const tap = new AudioWorkletNode(context, 'tap', { numberOfOutputs: 0 });   // nothing played back
    tap.port.onmessage = e => take(e.data);
    context.createMediaStreamSource(stream).connect(tap);
    return { context };
  }

  function take(chunk) {
    chunks.push(chunk);
    length += chunk.length;
    if (held === null) {   // not talking: keep the pre-roll only
      while (length - chunks[0].length >= PREROLL_S * rate) length -= chunks.shift().length;
      return;
    }
    for (const s of chunk) peak = Math.max(peak, Math.abs(s));
    if (length > (LONGEST_S + PREROLL_S) * rate) finish();
  }

  function tick() {
    button.style.setProperty('--level', Math.min(1, peak * 3).toFixed(2));
    button.textContent = `listening ${((performance.now() - held) / 1000).toFixed(1)} s`;
    peak = 0;
  }

  async function start() {
    if (held !== null) return;
    held = performance.now();
    button.classList.add('on');
    button.textContent = 'listening';
    if (!mic) {
      try {
        mic = await openMic();
      } catch (e) {
        held = null;
        reset();
        setState(`no microphone: ${e.message}`, true);
        return;
      }
    }
    mic.context.resume();
    if (held !== null && !ticker) ticker = setInterval(tick, 100);
  }

  function reset() {
    clearInterval(ticker);
    ticker = null;
    button.classList.remove('on');
    button.style.removeProperty('--level');
    button.textContent = 'Talk (hold V)';
  }

  async function finish() {
    if (held === null) return;
    const released = performance.now();
    const long = (released - held) / 1000;
    const audio = join(chunks, length);
    held = null;
    chunks = [];
    length = 0;
    reset();
    if (!mic || long < SHORTEST_S) {
      show({ heard: mic ? 'too short: hold, speak, then let go' : 'the microphone was opening: hold again and speak',
        command: '-', done: '-', timing: '-' });
      return;
    }
    show({ heard: 'hearing...', command: '...', done: '...', timing: '...' });
    try {
      const samples = await to16k(audio, rate);
      const reply = await fetch('/api/voice', { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: wav(samples) });
      const data = await reply.json();
      if (!reply.ok) throw new Error(data.fix ? `${data.error} -- ${data.fix}` : data.error || `HTTP ${reply.status}`);
      await carry(data, released);
    } catch (e) {
      show({ heard: '(not heard)', command: '-', done: e.message, error: true, timing: '-' });
      log(`voice: ${e.message}`, true);
    }
  }

  function join(parts, n) {
    const all = new Float32Array(n);
    let at = 0;
    for (const p of parts) { all.set(p, at); at += p.length; }
    return all;
  }

  async function to16k(samples, from) {   // the browser's own resampler, offline
    if (from === RATE) return samples;
    const offline = new OfflineAudioContext(1, Math.ceil(samples.length * RATE / from), RATE);
    const buffer = offline.createBuffer(1, samples.length, from);
    buffer.copyToChannel(samples, 0);
    const source = offline.createBufferSource();
    source.buffer = buffer;
    source.connect(offline.destination);
    source.start();
    return (await offline.startRendering()).getChannelData(0);
  }

  function wav(samples) {   // 16-bit PCM, mono, RATE
    const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
    const text = (at, s) => [...s].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)));
    text(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); text(8, 'WAVE');
    text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, RATE, true); view.setUint32(28, RATE * 2, true); view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, samples.length * 2, true);
    samples.forEach((s, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
    return view.buffer;
  }

  function describe(c) {
    const robot = { x3: 'X3', m3: 'M3', all: 'all' }[c.robot] ?? c.robot;
    if (c.action === 'stop') return 'STOP ALL';
    if (c.action === 'pick') {
      const what = [c.colour, c.size_mm && `${c.size_mm} mm`, c.kind ?? 'anything'].filter(Boolean).join(' ');
      return `pick, ${robot}: ${what}`;
    }
    if (c.action === 'unknown') return `not a command${c.why ? ` (${c.why})` : ''}`;
    return `${c.action.replace('_', ' ')}, ${robot}`;
  }

  async function carry(data, since) {
    // The command first (STOP above all), then what the page shows.
    const command = data.intent;
    const act = window.consoleActions?.[command.action];
    const doing = act ? (async () => act(command))() : null;   // called now; a throw is a rejection
    const ms = Math.round(performance.now() - since);
    const timing = `${ms} ms from ${data.asr_ms === undefined ? 'Enter' : 'letting go'} to the command` +
      (data.asr_ms === undefined ? '' : ` (${data.audio_s} s of speech, heard in ${data.asr_ms} ms)`);
    const heard = data.text ? `"${data.text}"` : '(nothing heard)';
    show({ heard, command: JSON.stringify(command), done: '...', timing });
    let done, error = false;
    try {
      if (doing) done = await doing;
      else if (command.action === 'unknown') done = 'nothing';
      else done = window.consoleActions ? `${command.action}: not wired yet` : 'the console is still starting';
    } catch (e) {
      done = e.message;
      error = true;
    }
    show({ done, error });
    remember(`${heard} -> ${describe(command)} -> ${done}  (${ms} ms)`, error);
    log(`voice: ${heard} -> ${describe(command)} -> ${done}`, error);
  }

  function show({ heard, command, done, timing, error = false }) {
    if (heard !== undefined) $('.heard', box).textContent = heard;
    if (command !== undefined) $('.command', box).textContent = command;
    if (done !== undefined) {
      $('.done', box).textContent = done;
      $('.done', box).classList.toggle('err', error);
    }
    if (timing !== undefined) $('.timing', box).textContent = timing;
  }

  function remember(text, error) {
    const item = document.createElement('li');
    item.textContent = `${new Date().toLocaleTimeString()}  ${text}`;
    if (error) item.className = 'err';
    $('.history', box).prepend(item);
    while ($('.history', box).childElementCount > KEPT) $('.history', box).lastElementChild.remove();
  }

  $('.typed', box).onsubmit = async e => {
    e.preventDefault();
    const input = $('input', e.target);
    const text = input.value.trim();
    if (!text) return;
    const since = performance.now();
    try {
      const reply = await fetch('/api/voice', { method: 'POST', headers: { 'Content-Type': 'text/plain; charset=utf-8' }, body: text });
      await carry(await reply.json(), since);
      input.value = '';
    } catch (e) {
      show({ done: e.message, error: true });
    }
  };

  button.addEventListener('pointerdown', e => { button.setPointerCapture(e.pointerId); start(); });
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(t => button.addEventListener(t, finish));
  addEventListener('keydown', e => {
    if (e.key.toLowerCase() !== KEY || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target instanceof Element && e.target.matches('input[type=text], textarea')) return;
    e.preventDefault();
    start();
  });
  addEventListener('keyup', e => { if (e.key.toLowerCase() === KEY) finish(); });
  // A key let go of in another window never comes back as a keyup: what was said is sent then.
  addEventListener('blur', finish);
  document.addEventListener('visibilitychange', () => { if (document.hidden) finish(); });
  status();
})();
