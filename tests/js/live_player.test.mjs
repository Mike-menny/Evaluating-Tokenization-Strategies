import assert from "node:assert/strict";
import test from "node:test";

import { createLivePlayer } from "../../dafx26_demo/static/js/live_player.mjs";

function midiNote({ pitch, velocity, start_sec, end_sec, event_id = 1 }) {
  return {
    name: "midi",
    data: {
      kind: "note",
      event_id,
      segment: 0,
      start_sec,
      end_sec,
      pitch,
      velocity,
    },
  };
}

function midiPedal({ value, time_sec, event_id = 2 }) {
  return {
    name: "midi",
    data: {
      kind: "pedal",
      event_id,
      segment: 0,
      time_sec,
      value,
    },
  };
}

function midiBeat({ beatKind = "beat", time_sec, event_id = 3 }) {
  return {
    name: "midi",
    data: {
      kind: "beat",
      event_id,
      segment: 0,
      time_sec,
      beat_kind: beatKind,
    },
  };
}

function sseBytes(name, data) {
  return new TextEncoder().encode(`event: ${name}\ndata: ${JSON.stringify(data)}\n\n`);
}

function fakeTone() {
  const callbacks = new Map();
  let nextId = 1;
  const rawContext = {
    sinkId: "",
    async setSinkId(id) {
      rawContext.sinkId = id;
    },
  };
  const tone = {
    lastFrequency: null,
    lastVelocity: null,
    synthDisposed: false,
    synthConnected: false,
    now: () => 0,
    start: async () => {},
    Frequency(pitch) {
      return { toFrequency: () => (Number(pitch) === 69 ? 440 : Number(pitch)) };
    },
    getContext() {
      return {
        rawContext,
        setTimeout(fn, delay, time) {
          const id = nextId++;
          callbacks.set(id, { fn, time });
          if (!(Number(delay) > 0)) {
            queueMicrotask(() => {
              const pending = callbacks.get(id);
              if (!pending) {
                return;
              }
              callbacks.delete(id);
              pending.fn(pending.time);
            });
          }
          return id;
        },
        clearTimeout(id) {
          callbacks.delete(id);
        },
      };
    },
    get pendingCallbacks() {
      return callbacks.size;
    },
    flush() {
      for (const [id, pending] of [...callbacks]) {
        callbacks.delete(id);
        pending.fn(pending.time);
      }
    },
  };
  tone.PolySynth = class {
    toDestination() {
      tone.synthConnected = true;
      return this;
    }
    triggerAttack(freq, _time, vel) {
      tone.lastFrequency = freq;
      tone.lastVelocity = vel;
    }
    triggerRelease() {}
    releaseAll() {}
    dispose() {
      tone.synthDisposed = true;
    }
  };
  return tone;
}

function fakeElements() {
  return {
    play: { disabled: false },
    stop: { disabled: false },
    mode: { value: "note", disabled: false, innerHTML: "", options: [] },
    device: { value: "", disabled: false, innerHTML: "" },
    indicator: { hidden: true, className: "" },
    indicatorText: { textContent: "" },
    speed: {
      value: "0.75",
      oninput: null,
      addEventListener(type, fn) {
        if (type === "input") {
          this.oninput = fn;
        }
      },
    },
    speedValue: { textContent: "" },
    roll: {
      notes: [],
      cleared: 0,
      started: 0,
      stopped: 0,
      addNote(note) {
        const index = this.notes.findIndex((item) => item.event_id === note.event_id);
        if (index >= 0) {
          this.notes[index] = note;
        } else {
          this.notes.push(note);
        }
      },
      pedals: [],
      addPedal(event) {
        const index = this.pedals.findIndex((item) => item.event_id === event.event_id);
        if (index >= 0) {
          this.pedals[index] = event;
        } else {
          this.pedals.push(event);
        }
      },
      beats: [],
      addBeat(event) {
        const index = this.beats.findIndex((item) => item.event_id === event.event_id);
        if (index >= 0) {
          this.beats[index] = event;
        } else {
          this.beats.push(event);
        }
      },
      setPlayhead() {},
      setPedal() {},
      clear() {
        this.cleared += 1;
        this.notes = [];
        this.pedals = [];
        this.beats = [];
      },
      start() {
        this.started += 1;
      },
      stop() {
        this.stopped += 1;
      },
    },
  };
}

function fakePiano() {
  const piano = {
    down: [],
    up: [],
    playNoteDown(note) {
      piano.down.push(note);
    },
    playNoteUp(note) {
      piano.up.push(note);
    },
  };
  return piano;
}

function makePlayer({ tone, network, elements, setStatus, piano, mediaDevices } = {}) {
  const resolvedPiano = piano ?? fakePiano();
  return {
    piano: resolvedPiano,
    player: createLivePlayer({
      Tone: tone ?? fakeTone(),
      fetchImpl: network.fetch,
      elements: elements ?? fakeElements(),
      getRequest: () => ({ mode: "note", composer: "Chopin", genre: "etude", seed: 0 }),
      setStatus: setStatus ?? (() => {}),
      mediaDevices,
      createPiano: async () => resolvedPiano,
    }),
  };
}

function fakeMediaDevices(outputs) {
  return {
    enumerateDevices: async () => outputs,
  };
}

function deferredStopFetch() {
  const encoder = new TextEncoder();
  let controller;
  const stream = new ReadableStream({
    start(c) {
      controller = c;
    },
  });
  let startedResolve;
  const network = {
    started: new Promise((resolve) => {
      startedResolve = resolve;
    }),
    stopRequestSettled: false,
    lastLiveBody: null,
    emit(event) {
      controller.enqueue(sseBytes(event.name, event.data));
    },
    emitBytes(bytes) {
      controller.enqueue(bytes);
    },
    close() {
      controller.close();
    },
    fetch(url, init = {}) {
      if (String(url).includes("/stop")) {
        return new Promise((resolve) => {
          network._settleStop = () => {
            network.stopRequestSettled = true;
            resolve({ ok: true, json: async () => ({ ok: true }) });
          };
        });
      }
      network.lastLiveBody = JSON.parse(init.body);
      startedResolve();
      return Promise.resolve({
        ok: true,
        body: stream,
      });
    },
  };
  return network;
}

test("player converts MIDI units and Stop is locally immediate", async () => {
  const tone = fakeTone();
  const network = deferredStopFetch();
  const piano = fakePiano();
  const { player } = makePlayer({ tone, network, piano });
  const running = player.play();
  await network.started;
  network.emit({ name: "meta", data: { job_id: "job1", live_buffer_sec: 2 } });
  network.emit(midiNote({ pitch: 69, velocity: 64, start_sec: 1, end_sec: 2 }));
  await new Promise((resolve) => setTimeout(resolve, 10));
  tone.flush();
  assert.equal(piano.down[0].pitch, 69);
  assert.equal(piano.down[0].velocity, 64);
  assert.equal(network.lastLiveBody.mode, "note");
  assert.equal("max_tokens" in network.lastLiveBody, false);
  assert.equal(network.lastLiveBody.temperature, 0.95);
  assert.equal(network.lastLiveBody.top_p, 0.98);
  player.stop();
  assert.ok(piano.up.length > 0);
  assert.equal(tone.pendingCallbacks, 0);
  assert.equal(network.stopRequestSettled, false);
  network.close();
  await running;
});

test("fragmented byte streams still schedule notes", async () => {
  const tone = fakeTone();
  const network = deferredStopFetch();
  const piano = fakePiano();
  const { player } = makePlayer({
    tone,
    network,
    piano,
    setStatus: () => {},
  });
  const running = player.play();
  await network.started;
  const frame = sseBytes("meta", { job_id: "j2", live_buffer_sec: 2 });
  network.emitBytes(frame.slice(0, 12));
  network.emitBytes(frame.slice(12));
  const note = sseBytes("midi", midiNote({ pitch: 60, velocity: 127, start_sec: 0, end_sec: 1 }).data);
  network.emitBytes(note.slice(0, 8));
  network.emitBytes(note.slice(8));
  await new Promise((resolve) => setTimeout(resolve, 10));
  tone.flush();
  assert.equal(piano.down[0].velocity, 127);
  player.stop();
  network.close();
  await running;
});

test("non-MLX meta disables live controls", () => {
  const elements = fakeElements();
  const player = createLivePlayer({
    Tone: fakeTone(),
    fetchImpl: async () => ({ ok: true, body: new ReadableStream() }),
    elements,
    getRequest: () => ({ mode: "note", composer: "Chopin", genre: "etude", seed: 0 }),
    setStatus: () => {},
  });
  player.syncMeta({ backend: "pytorch", modes: ["note", "full"] });
  assert.equal(elements.play.disabled, true);
  assert.equal(elements.stop.disabled, true);
  assert.equal(elements.mode.disabled, true);
});

test("lists the current audio output", async () => {
  const tone = fakeTone();
  const network = deferredStopFetch();
  const elements = fakeElements();
  const { player } = makePlayer({
    tone,
    network,
    elements,
    mediaDevices: fakeMediaDevices([
      { kind: "audioinput", deviceId: "mic", label: "Mic" },
      { kind: "audiooutput", deviceId: "", label: "MacBook Speakers" },
      { kind: "audiooutput", deviceId: "airpods", label: "AirPods" },
    ]),
  });
  const running = player.play();
  await network.started;
  assert.match(elements.device.innerHTML, /MacBook Speakers/);
  assert.match(elements.device.innerHTML, /AirPods/);
  assert.equal(elements.device.value, "");
  elements.device.value = "airpods";
  await elements.device.onchange();
  assert.equal(tone.getContext().rawContext.sinkId, "airpods");
  player.stop();
  network.close();
  await running;
});

test("speeding up does not place later notes over already queued music", async () => {
  const tone = fakeTone();
  const network = deferredStopFetch();
  const elements = fakeElements();
  const { player } = makePlayer({ tone, network, elements });
  const running = player.play();
  await network.started;
  network.emit({ name: "meta", data: { job_id: "job1", live_buffer_sec: 2 } });
  network.emit(midiNote({ event_id: 1, pitch: 60, velocity: 64, start_sec: 10, end_sec: 11 }));
  await new Promise((resolve) => setTimeout(resolve, 10));
  elements.speed.value = "1";
  elements.speed.oninput();
  network.emit(midiNote({ event_id: 2, pitch: 62, velocity: 64, start_sec: 12, end_sec: 13 }));
  await new Promise((resolve) => setTimeout(resolve, 10));
  const [first, second] = elements.roll.notes;
  assert.equal(first.start_sec, 10);
  assert.equal(second.start_sec, 12);
  assert.equal(first.startAudio, 2 + 10);
  assert.equal(second.startAudio, 2 + 12);
  assert.ok(second.startAudio > first.startAudio);
  player.stop();
  network.close();
  await running;
});

test("live notes stretch to the speed slider", async () => {
  const tone = fakeTone();
  const network = deferredStopFetch();
  const elements = fakeElements();
  elements.speed.value = "0.75";
  const { player } = makePlayer({ tone, network, elements });
  const running = player.play();
  await network.started;
  network.emit({ name: "meta", data: { job_id: "job1", live_buffer_sec: 2 } });
  network.emit(midiNote({ pitch: 64, velocity: 90, start_sec: 1.5, end_sec: 2.1 }));
  await new Promise((resolve) => setTimeout(resolve, 10));
  const note = elements.roll.notes[0];
  assert.equal(note.startAudio, 2 + 1.5 / 0.75);
  assert.equal(note.endAudio, 2 + 1.5 / 0.75 + 0.6 / 0.75);
  player.stop();
  network.close();
  await running;
});

test("live piano roll receives notes and clears on stop", async () => {
  const tone = fakeTone();
  const network = deferredStopFetch();
  const elements = fakeElements();
  const { player } = makePlayer({ tone, network, elements });
  const running = player.play();
  await network.started;
  network.emit({ name: "meta", data: { job_id: "job1", live_buffer_sec: 2 } });
  network.emit(midiNote({ pitch: 64, velocity: 90, start_sec: 1, end_sec: 2 }));
  network.emit(midiPedal({ event_id: 2, value: 127, time_sec: 1.2 }));
  network.emit(midiBeat({ event_id: 3, beatKind: "downbeat", time_sec: 1.5 }));
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(elements.roll.notes[0].pitch, 64);
  assert.equal(elements.roll.pedals[0].value, 127);
  assert.equal(elements.roll.pedals[0].timeAudio, 2 + 1.2 / 0.75);
  assert.equal(elements.roll.beats[0].beatKind, "downbeat");
  assert.equal(elements.roll.beats[0].timeAudio, 2 + 1.5 / 0.75);
  assert.ok(elements.roll.started > 0);
  player.stop();
  assert.ok(elements.roll.stopped > 0);
  assert.equal(elements.roll.notes.length, 0);
  assert.equal(elements.roll.pedals.length, 0);
  assert.equal(elements.roll.beats.length, 0);
  network.close();
  await running;
});

test("shows buffering until the preroll elapses", async () => {
  const tone = fakeTone();
  const network = deferredStopFetch();
  const elements = fakeElements();
  const { player } = makePlayer({ tone, network, elements });
  const running = player.play();
  await network.started;
  assert.equal(elements.indicator.hidden, false);
  assert.match(elements.indicator.className, /is-loading/);
  network.emit({ name: "meta", data: { job_id: "job1", live_buffer_sec: 2 } });
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.match(elements.indicator.className, /is-buffering/);
  assert.match(elements.indicatorText.textContent, /Buffering/i);
  tone.flush();
  assert.match(elements.indicator.className, /is-playing/);
  player.stop();
  assert.equal(elements.indicator.hidden, true);
  network.close();
  await running;
});

test("error and end events update status and second play works", async () => {
  const statuses = [];
  const network = deferredStopFetch();
  const { player } = makePlayer({
    network,
    setStatus: (text) => statuses.push(text),
  });
  player.syncMeta({ backend: "mlx", modes: ["note"] });
  const first = player.play();
  await network.started;
  network.emit({ name: "end", data: { reason: "error", message: "boom" } });
  network.close();
  await first;
  assert.ok(statuses.some((text) => String(text).includes("boom")));

  const network2 = deferredStopFetch();
  const { player: player2 } = makePlayer({ network: network2 });
  const second = player2.play();
  await network2.started;
  network2.emit({ name: "meta", data: { job_id: "j3", live_buffer_sec: 2 } });
  player2.stop();
  network2.close();
  await second;
});
