import assert from "node:assert/strict";
import test from "node:test";

import {
  createPedalState,
  createPlaybackClock,
  playbackSpeed,
  scheduleNote,
  schedulePedal,
  stopState,
} from "../../dafx26_demo/static/js/live_schedule.mjs";

test("late note shifts and preserves duration", () => {
  const note = scheduleNote(
    { start_sec: 1, end_sec: 3, pitch: 60, velocity: 64 },
    10,
    0,
  );
  assert.equal(note.startAudio, 10.05);
  assert.equal(note.endAudio, 12.05);
  assert.equal(note.underrun, true);
});

test("on-time note keeps model time", () => {
  const note = scheduleNote(
    { start_sec: 1, end_sec: 1.5, pitch: 64, velocity: 80 },
    0.2,
    0,
  );
  assert.equal(note.startAudio, 1);
  assert.equal(note.endAudio, 1.5);
  assert.equal(note.underrun, false);
});

test("0.75x speed stretches onset and duration", () => {
  const note = scheduleNote(
    { start_sec: 1.5, end_sec: 2.1, pitch: 60, velocity: 64 },
    0,
    2,
    0.75,
  );
  assert.equal(note.startAudio, 2 + 1.5 / 0.75);
  assert.equal(note.endAudio, 2 + 1.5 / 0.75 + 0.6 / 0.75);
  assert.equal(note.underrun, false);
});

test("speeding up keeps later music after earlier music", () => {
  const clock = createPlaybackClock({ epoch: 2, speed: 0.75 });
  const first = scheduleNote(
    { start_sec: 10, end_sec: 11, pitch: 60, velocity: 64 },
    0,
    clock,
  );
  clock.setSpeed(1, 0);
  const firstAgain = scheduleNote(
    { start_sec: 10, end_sec: 11, pitch: 60, velocity: 64 },
    0,
    clock,
  );
  const second = scheduleNote(
    { start_sec: 12, end_sec: 13, pitch: 62, velocity: 64 },
    0,
    clock,
  );
  assert.equal(first.startAudio, 2 + 10 / 0.75);
  assert.equal(firstAgain.startAudio, 2 + 10);
  assert.equal(second.startAudio, 2 + 12);
  assert.ok(second.startAudio > firstAgain.startAudio);
});

test("a mid-play speed change continues from the current music position", () => {
  const clock = createPlaybackClock({ epoch: 2, speed: 0.75 });
  clock.setSpeed(1, 4);
  assert.equal(clock.musicAt(4), (4 - 2) * 0.75);
  assert.equal(clock.audioAt(3), 4 + (3 - 1.5) / 1);
  assert.ok(clock.audioAt(4) > clock.audioAt(3));
});

test("playbackSpeed clamps between 0.5 and 1 and defaults to 0.75", () => {
  assert.equal(playbackSpeed(), 0.75);
  assert.equal(playbackSpeed("0.75"), 0.75);
  assert.equal(playbackSpeed("0.5"), 0.5);
  assert.equal(playbackSpeed("0.55"), 0.55);
  assert.equal(playbackSpeed("1"), 1);
  assert.equal(playbackSpeed("0.73"), 0.75);
  assert.equal(playbackSpeed("0.1"), 0.5);
  assert.equal(playbackSpeed("1.4"), 1);
  assert.equal(playbackSpeed("nope"), 0.75);
});

test("late pedal is shifted by 50ms", () => {
  const pedal = schedulePedal({ time_sec: 0.5, value: 127 }, 3, 0);
  assert.equal(pedal.timeAudio, 3.05);
  assert.equal(pedal.underrun, true);
});

test("pedal state changes only when the scheduled callback fires", () => {
  const state = createPedalState();
  const note = { event_id: 1, pitch: 60 };
  state.schedulePedalChange(127, () => {});
  assert.equal(state.down, false);
  state.firePedal(127);
  assert.equal(state.down, true);
  const release = state.noteOff(note);
  assert.equal(release.deferred, true);
  assert.equal(state.activeIds().includes(1), true);
  const flushed = state.firePedal(0);
  assert.equal(state.down, false);
  assert.deepEqual(flushed.map((item) => item.event_id), [1]);
});

test("overlapping same-pitch notes are tracked by event id", () => {
  const state = createPedalState();
  state.noteOn({ event_id: 7, pitch: 60 });
  state.noteOn({ event_id: 8, pitch: 60 });
  assert.deepEqual(state.activeIds().sort(), [7, 8]);
  const first = state.noteOff({ event_id: 7, pitch: 60 });
  assert.equal(first.deferred, false);
  assert.deepEqual(state.activeIds(), [8]);
});

test("stop clears future, deferred, and active notes", () => {
  const state = createPedalState();
  state.noteOn({ event_id: 3, pitch: 64 });
  state.firePedal(127);
  state.noteOff({ event_id: 4, pitch: 65 });
  state.scheduleFuture(99);
  const stopped = stopState(state);
  assert.deepEqual(stopped.activeIds.sort(), [3, 4]);
  assert.deepEqual(stopped.futureIds, [99]);
  assert.equal(state.down, false);
  assert.deepEqual(state.activeIds(), []);
});
