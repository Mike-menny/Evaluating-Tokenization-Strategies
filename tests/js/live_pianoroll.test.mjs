import assert from "node:assert/strict";
import test from "node:test";

import {
  AHEAD,
  BEHIND,
  PLAYHEAD_FRACTION,
  isBlackKey,
  layoutBeat,
  layoutNote,
  layoutPedal,
  pedalSpans,
} from "../../dafx26_demo/static/js/live_pianoroll.mjs";

const view = {
  playhead: 10,
  width: 200,
  height: 88,
  behind: BEHIND,
  ahead: AHEAD,
  minPitch: 21,
  maxPitch: 108,
  gutter: 0,
};

test("playhead sits in the left fifth of the roll", () => {
  assert.ok(PLAYHEAD_FRACTION >= 0.15 && PLAYHEAD_FRACTION <= 0.2);
  assert.ok(Math.abs(BEHIND / (BEHIND + AHEAD) - PLAYHEAD_FRACTION) < 1e-12);
});

test("future notes sit to the right of the playhead", () => {
  const playX = 200 * PLAYHEAD_FRACTION;
  const rect = layoutNote({ startAudio: 11, endAudio: 12, pitch: 108 }, view);
  assert.ok(rect.x > playX);
  assert.equal(rect.active, false);
  assert.equal(rect.y, 0);
});

test("sounding notes are marked active", () => {
  const rect = layoutNote({ startAudio: 9.5, endAudio: 10.5, pitch: 21 }, view);
  assert.equal(rect.active, true);
  assert.ok(rect.y > 80);
});

test("black keys are the piano accidentals", () => {
  assert.equal(isBlackKey(60), false);
  assert.equal(isBlackKey(61), true);
});

test("pedal spans become closed bars and stay open until release", () => {
  const closed = pedalSpans(
    [
      { timeAudio: 9, value: 127 },
      { timeAudio: 11, value: 0 },
    ],
    10,
    AHEAD,
  );
  assert.deepEqual(closed, [{ startAudio: 9, endAudio: 11 }]);
  const open = pedalSpans([{ timeAudio: 12, value: 80 }], 10, AHEAD);
  assert.equal(open[0].startAudio, 12);
  assert.equal(open[0].endAudio, 10 + AHEAD);
});

test("pedal bar sits in a bottom lane in a different color", () => {
  const pedalView = { ...view, pedalLane: 8 };
  const pedal = layoutPedal({ startAudio: 9.5, endAudio: 11 }, pedalView);
  assert.equal(pedal.y, 88 - 8);
  assert.equal(pedal.h, 8);
  assert.equal(pedal.y + pedal.h, pedalView.height);
  assert.equal(pedal.active, true);
  assert.match(pedal.fill, /90,\s*176,\s*196/);
});

test("beats are small green blips above the pedal lane", () => {
  const beat = layoutBeat({ timeAudio: 11, beatKind: "beat" }, view);
  const downbeat = layoutBeat({ timeAudio: 11, beatKind: "downbeat" }, view);
  assert.ok(beat.x > 200 * PLAYHEAD_FRACTION);
  assert.equal(beat.w, 2);
  assert.ok(beat.y + beat.h <= view.height - 8);
  assert.ok(downbeat.h > beat.h);
  assert.match(beat.fill, /125,\s*200,\s*118/);
});
