import assert from "node:assert/strict";
import test from "node:test";

import { createSampledPiano } from "../../dafx26_demo/static/js/live_piano.mjs";

function fakeTone() {
  const sources = [];
  const reverbs = [];
  const tone = {
    sources,
    reverbs,
    now: () => 1.25,
    loaded: async () => {},
    ToneAudioBuffers: class {
      constructor(urls) {
        this.urls = urls;
      }
      get(name) {
        return { id: name };
      }
    },
    Reverb: class {
      constructor(opts) {
        this.opts = opts;
        this.ready = Promise.resolve(this);
        reverbs.push(this);
      }
      toDestination() {
        this.destination = true;
        return this;
      }
    },
    ToneBufferSource: class {
      constructor(opts) {
        this.opts = opts;
        this.started = null;
        this.stopped = null;
        this.connected = null;
        sources.push(this);
      }
      connect(node) {
        this.connected = node;
        return this;
      }
      start(time, offset, duration, gain) {
        this.started = { time, offset, duration, gain };
        return this;
      }
      stop(time) {
        this.stopped = time;
        return this;
      }
    },
  };
  return tone;
}

test("overlapping same-pitch notes keep independent voices", async () => {
  const Tone = fakeTone();
  const piano = await createSampledPiano(Tone);
  piano.playNoteDown({ event_id: 1, pitch: 60, velocity: 80, time: 1 });
  piano.playNoteDown({ event_id: 2, pitch: 60, velocity: 100, time: 1.1 });
  assert.equal(Tone.sources.length, 2);
  piano.playNoteUp({ event_id: 1, time: 2 });
  assert.equal(Tone.sources[0].stopped, 2);
  assert.equal(Tone.sources[1].stopped, null);
});

test("attacks fade and route through a little reverb", async () => {
  const Tone = fakeTone();
  const piano = await createSampledPiano(Tone);
  piano.playNoteDown({ event_id: 3, pitch: 69, velocity: 64, time: 0.5 });
  assert.ok(Tone.sources[0].opts.fadeIn > 0);
  assert.ok(Tone.sources[0].opts.fadeOut > 0);
  assert.equal(Tone.sources[0].connected, Tone.reverbs[0]);
  assert.equal(Tone.reverbs[0].destination, true);
  assert.ok(Tone.reverbs[0].opts.wet > 0);
  assert.ok(Tone.reverbs[0].opts.wet < 0.35);
  assert.equal(Tone.sources[0].started.gain, 64 / 127);
});
