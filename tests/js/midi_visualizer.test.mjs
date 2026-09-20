import assert from "node:assert/strict";
import test from "node:test";

import {
  PIANO_ROLL_CONFIG,
  beatMarkerPositions,
  bindPlayerVisualizer,
} from "../../dafx26_demo/static/js/midi_visualizer.mjs";

test("bindPlayerVisualizer uses light notes and syncs on load", () => {
  const player = {
    noteSequence: null,
    listeners: {},
    addEventListener(name, fn) {
      this.listeners[name] = fn;
    },
  };
  const visualizer = { config: null, noteSequence: null };
  bindPlayerVisualizer(player, visualizer);
  assert.equal(visualizer.config.noteRGB, PIANO_ROLL_CONFIG.noteRGB);
  assert.match(visualizer.config.noteRGB, /196/);
  player.noteSequence = { notes: [{ pitch: 60 }] };
  player.listeners.load();
  assert.equal(visualizer.noteSequence, player.noteSequence);
});

test("comparison beat markers become green strip positions", () => {
  const markers = beatMarkerPositions(
    [
      { time_sec: 1, kind: "beat" },
      { time_sec: 2, kind: "downbeat" },
    ],
    4,
  );
  assert.deepEqual(markers, [
    { leftPercent: 25, kind: "beat" },
    { leftPercent: 50, kind: "downbeat" },
  ]);
  assert.deepEqual(beatMarkerPositions([], 4), []);
});
