const MIN_PITCH = 21;
const MAX_PITCH = 108;
export const PLAYHEAD_FRACTION = 0.18;
export const BEHIND = 2;
export const AHEAD = BEHIND * (1 - PLAYHEAD_FRACTION) / PLAYHEAD_FRACTION;
const GUTTER = 28;
export const PEDAL_LANE = 8;
export const PEDAL_RGB = "90, 176, 196";
export const BEAT_RGB = "125, 200, 118";

export function isBlackKey(pitch) {
  return [1, 3, 6, 8, 10].includes(((Number(pitch) % 12) + 12) % 12);
}

export function layoutNote(note, view) {
  const behind = view.behind ?? BEHIND;
  const ahead = view.ahead ?? AHEAD;
  const gutter = view.gutter ?? GUTTER;
  const minPitch = view.minPitch ?? MIN_PITCH;
  const maxPitch = view.maxPitch ?? MAX_PITCH;
  const span = behind + ahead;
  const start = view.playhead - behind;
  const usable = Math.max(1, view.width - gutter);
  const x = gutter + ((note.startAudio - start) / span) * usable;
  const w = Math.max(2, ((note.endAudio - note.startAudio) / span) * usable);
  const pitches = maxPitch - minPitch + 1;
  const h = view.height / pitches;
  const y = (maxPitch - note.pitch) * h;
  const active = view.playhead >= note.startAudio && view.playhead < note.endAudio;
  return { x, y, w, h, active };
}

export function pedalSpans(events, playhead, ahead = AHEAD) {
  const ordered = [...events].sort((a, b) => a.timeAudio - b.timeAudio);
  const spans = [];
  let start = null;
  for (const event of ordered) {
    const down = Number(event.value) >= 64;
    if (down && start == null) {
      start = event.timeAudio;
    } else if (!down && start != null) {
      spans.push({ startAudio: start, endAudio: event.timeAudio });
      start = null;
    }
  }
  if (start != null) {
    spans.push({ startAudio: start, endAudio: Math.max(Number(playhead) + ahead, start) });
  }
  return spans;
}

export function layoutPedal(span, view) {
  const behind = view.behind ?? BEHIND;
  const ahead = view.ahead ?? AHEAD;
  const gutter = view.gutter ?? GUTTER;
  const lane = view.pedalLane ?? PEDAL_LANE;
  const windowSec = behind + ahead;
  const start = view.playhead - behind;
  const usable = Math.max(1, view.width - gutter);
  const x = gutter + ((span.startAudio - start) / windowSec) * usable;
  const w = Math.max(2, ((span.endAudio - span.startAudio) / windowSec) * usable);
  const y = view.height - lane;
  const active = view.playhead >= span.startAudio && view.playhead < span.endAudio;
  const alpha = active ? 0.85 : 0.45;
  return {
    x,
    y,
    w,
    h: lane,
    active,
    fill: `rgba(${PEDAL_RGB}, ${alpha})`,
  };
}

export function layoutBeat(beat, view) {
  const behind = view.behind ?? BEHIND;
  const ahead = view.ahead ?? AHEAD;
  const gutter = view.gutter ?? GUTTER;
  const pedalLane = view.pedalLane ?? PEDAL_LANE;
  const windowSec = behind + ahead;
  const start = view.playhead - behind;
  const usable = Math.max(1, view.width - gutter);
  const downbeat = beat.beatKind === "downbeat";
  const h = downbeat ? 7 : 4;
  return {
    x: gutter + ((beat.timeAudio - start) / windowSec) * usable,
    y: view.height - pedalLane - h,
    w: 2,
    h,
    fill: `rgba(${BEAT_RGB}, ${downbeat ? 0.95 : 0.75})`,
  };
}

export function createPianoRoll(canvas, { now, requestFrame, cancelFrame } = {}) {
  const notes = new Map();
  const pedals = new Map();
  const beats = new Map();
  let playhead = 0;
  let frame = 0;
  const getNow = now ?? (() => 0);
  const request = requestFrame ?? (typeof requestAnimationFrame === "function" ? requestAnimationFrame : () => 0);
  const cancel = cancelFrame ?? (typeof cancelAnimationFrame === "function" ? cancelAnimationFrame : () => {});

  function size() {
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    const width = Math.max(1, canvas.clientWidth || canvas.width || 1);
    const height = Math.max(1, canvas.clientHeight || canvas.height || 1);
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    return { width, height, dpr };
  }

  function draw() {
    const ctx = canvas.getContext?.("2d");
    if (!ctx) {
      return;
    }
    const { width, height, dpr } = size();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#100e0c";
    ctx.fillRect(0, 0, width, height);
    const pitches = MAX_PITCH - MIN_PITCH + 1;
    const row = height / pitches;
    for (let pitch = MIN_PITCH; pitch <= MAX_PITCH; pitch += 1) {
      const y = (MAX_PITCH - pitch) * row;
      ctx.fillStyle = isBlackKey(pitch) ? "#16130f" : "#1b1814";
      ctx.fillRect(0, y, width, row);
    }
    ctx.fillStyle = "#221e18";
    ctx.fillRect(0, 0, GUTTER, height);
    for (let pitch = MIN_PITCH; pitch <= MAX_PITCH; pitch += 1) {
      const y = (MAX_PITCH - pitch) * row;
      ctx.fillStyle = isBlackKey(pitch) ? "#3d352c" : "#f4eee4";
      ctx.fillRect(4, y + 0.5, GUTTER - 8, Math.max(1, row - 1));
    }
    const view = { playhead, width, height, behind: BEHIND, ahead: AHEAD, minPitch: MIN_PITCH, maxPitch: MAX_PITCH, gutter: GUTTER, pedalLane: PEDAL_LANE };
    for (const note of notes.values()) {
      const rect = layoutNote(note, view);
      if (rect.x + rect.w < 0 || rect.x > width) {
        continue;
      }
      const velocity = Math.max(0.25, Math.min(1, Number(note.velocity ?? 80) / 127));
      ctx.fillStyle = rect.active ? `rgba(244, 238, 228, ${0.55 + 0.45 * velocity})` : `rgba(196, 163, 90, ${0.35 + 0.5 * velocity})`;
      ctx.fillRect(rect.x, rect.y + 0.4, rect.w, Math.max(1.5, rect.h - 0.8));
    }
    for (const span of pedalSpans(pedals.values(), playhead, AHEAD)) {
      const rect = layoutPedal(span, view);
      if (rect.x + rect.w < 0 || rect.x > width) {
        continue;
      }
      ctx.fillStyle = rect.fill;
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    }
    for (const beat of beats.values()) {
      const rect = layoutBeat(beat, view);
      if (rect.x + rect.w < 0 || rect.x > width) {
        continue;
      }
      ctx.fillStyle = rect.fill;
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    }
    const playX = GUTTER + (BEHIND / (BEHIND + AHEAD)) * (width - GUTTER);
    ctx.strokeStyle = "#c4a35a";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, height);
    ctx.stroke();
  }

  function tick() {
    playhead = getNow();
    draw();
    frame = request(tick);
  }

  function recordPedal(event) {
    pedals.set(event.event_id, event);
    draw();
  }

  return {
    addNote(note) {
      notes.set(note.event_id, note);
      draw();
    },
    setPlayhead(time) {
      playhead = time;
      draw();
    },
    addPedal(event) {
      recordPedal(event);
    },
    addBeat(event) {
      beats.set(event.event_id, event);
      draw();
    },
    setPedal(down, time) {
      recordPedal({
        event_id: down ? "pedal-down" : "pedal-up",
        timeAudio: time ?? playhead,
        value: down ? 127 : 0,
      });
    },
    clear() {
      notes.clear();
      pedals.clear();
      beats.clear();
      draw();
    },
    start() {
      if (frame) {
        return;
      }
      tick();
    },
    stop() {
      if (frame) {
        cancel(frame);
        frame = 0;
      }
    },
  };
}
