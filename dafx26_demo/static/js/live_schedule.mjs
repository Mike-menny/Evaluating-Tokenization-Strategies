export function playbackSpeed(value, fallback = 0.75) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return fallback;
  }
  return Math.min(1, Math.max(0.5, Math.round(n * 20) / 20));
}

function playbackRate(speed) {
  const rate = Number(speed);
  return rate > 0 ? rate : 1;
}

export function createPlaybackClock({ epoch = 0, speed = 1 } = {}) {
  let originMusic = 0;
  let originAudio = Number(epoch) || 0;
  let rate = playbackRate(speed);

  function musicAt(audioTime) {
    return originMusic + (Number(audioTime) - originAudio) * rate;
  }

  function audioAt(musicSec) {
    return originAudio + (Number(musicSec) - originMusic) / rate;
  }

  function setSpeed(speed, audioNow) {
    const next = playbackRate(playbackSpeed(speed, rate));
    if (next === rate) {
      return;
    }
    const musicZeroAudio = audioAt(0);
    if (Number(audioNow) < musicZeroAudio) {
      originMusic = 0;
      originAudio = musicZeroAudio;
      rate = next;
      return;
    }
    originMusic = musicAt(audioNow);
    originAudio = Number(audioNow);
    rate = next;
  }

  return {
    musicAt,
    audioAt,
    setSpeed,
    get speed() {
      return rate;
    },
  };
}

function asClock(epochOrClock, speed) {
  if (epochOrClock && typeof epochOrClock.audioAt === "function") {
    return epochOrClock;
  }
  return createPlaybackClock({ epoch: epochOrClock, speed: speed ?? 1 });
}

export function scheduleNote(event, audioNow, epochOrClock, speed = 1) {
  const clock = asClock(epochOrClock, speed);
  const planned = clock.audioAt(event.start_sec);
  const duration = Math.max(0.01, (event.end_sec - event.start_sec) / clock.speed);
  const startAudio = planned < audioNow ? audioNow + 0.05 : planned;
  return {
    ...event,
    startAudio,
    endAudio: startAudio + duration,
    underrun: planned < audioNow,
  };
}

export function schedulePedal(event, audioNow, epochOrClock, speed = 1) {
  const planned = asClock(epochOrClock, speed).audioAt(event.time_sec);
  const timeAudio = planned < audioNow ? audioNow + 0.05 : planned;
  return {
    ...event,
    timeAudio,
    underrun: planned < audioNow,
  };
}

export function createPedalState() {
  const active = new Map();
  const deferred = [];
  const future = [];
  let down = false;

  return {
    get down() {
      return down;
    },
    schedulePedalChange() {
      return down;
    },
    firePedal(value) {
      down = Number(value) >= 64;
      if (down) {
        return [];
      }
      const flushed = deferred.splice(0);
      for (const note of flushed) {
        active.delete(note.event_id);
      }
      return flushed;
    },
    noteOn(note) {
      active.set(note.event_id, note);
    },
    noteOff(note) {
      if (!active.has(note.event_id)) {
        active.set(note.event_id, note);
      }
      if (down) {
        deferred.push(note);
        return { deferred: true, note };
      }
      active.delete(note.event_id);
      return { deferred: false, note };
    },
    activeIds() {
      const ids = new Set([...active.keys(), ...deferred.map((note) => note.event_id)]);
      return [...ids];
    },
    scheduleFuture(id) {
      future.push(id);
    },
    futureIds() {
      return [...future];
    },
    reset() {
      down = false;
      active.clear();
      deferred.length = 0;
      future.length = 0;
    },
  };
}

export function stopState(state) {
  const activeIds = [...state.activeIds()].sort((a, b) => a - b);
  const futureIds = [...state.futureIds()];
  state.reset();
  return { activeIds, futureIds };
}
