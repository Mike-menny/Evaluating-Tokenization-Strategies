import { createSampledPiano } from "./live_piano.mjs";
import { createPianoRoll } from "./live_pianoroll.mjs";
import { createPedalState, createPlaybackClock, playbackSpeed, scheduleNote, schedulePedal, stopState } from "./live_schedule.mjs";
import { createSseParser } from "./sse_parse.mjs";

function clampMidiVelocity(velocity) {
  return Math.max(0, Math.min(127, Number(velocity)));
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch],
  );
}

function outputLabel(device) {
  if (device.label) {
    return device.label;
  }
  if (!device.deviceId || device.deviceId === "default") {
    return "Default system output";
  }
  return `Output ${device.deviceId.slice(0, 8)}`;
}

export async function loadSoundFontPiano(Tone) {
  return createSampledPiano(Tone);
}

export function createLivePlayer({
  Tone,
  fetchImpl,
  elements,
  getRequest,
  setStatus,
  mediaDevices,
  createPiano,
}) {
  const devicesApi = mediaDevices ?? (typeof navigator !== "undefined" ? navigator.mediaDevices : null);
  const loadPiano =
    createPiano ??
    (() => loadSoundFontPiano(Tone));
  let controller = null;
  let piano = null;
  let cancelled = false;
  let jobId = null;
  let clock = null;
  let underruns = 0;
  let scheduledIds = [];
  let queued = [];
  let pedal = createPedalState();
  const sounding = new Map();
  const roll =
    elements.roll && typeof elements.roll.addNote === "function"
      ? elements.roll
      : elements.roll
        ? createPianoRoll(elements.roll, { now: () => Tone.now() })
        : null;

  function setControlsEnabled(enabled) {
    elements.play.disabled = !enabled;
    elements.stop.disabled = !enabled;
    if (elements.mode) {
      elements.mode.disabled = !enabled;
    }
  }

  function currentSpeed() {
    return playbackSpeed(elements.speed?.value);
  }

  function renderSpeedLabel() {
    if (elements.speedValue) {
      elements.speedValue.textContent = `${currentSpeed()}×`;
    }
  }

  function applyPlaybackSpeed() {
    renderSpeedLabel();
    if (!clock) {
      return;
    }
    clock.setSpeed(currentSpeed(), Tone.now());
    rescheduleQueued();
  }

  if (elements.speed && typeof elements.speed.addEventListener === "function") {
    elements.speed.addEventListener("input", applyPlaybackSpeed);
  }
  renderSpeedLabel();

  function setLiveState(state, text) {
    if (!elements.indicator) {
      return;
    }
    elements.indicator.hidden = state === "idle";
    elements.indicator.className = `live-indicator is-${state}`;
    if (elements.indicatorText) {
      elements.indicatorText.textContent = text;
    }
  }

  function clearScheduled() {
    const context = Tone.getContext();
    for (const id of scheduledIds) {
      context.clearTimeout(id);
    }
    scheduledIds = [];
  }

  function scheduleAt(when, fn) {
    const delay = Math.max(0, when - Tone.now());
    const id = Tone.getContext().setTimeout((time) => fn(time), delay, when);
    scheduledIds.push(id);
    pedal.scheduleFuture(id);
    return id;
  }

  async function refreshOutputDevice() {
    if (!elements.device) {
      return;
    }
    const sinkId = Tone.getContext()?.rawContext?.sinkId ?? "";
    let outputs = [];
    try {
      outputs = devicesApi ? (await devicesApi.enumerateDevices()).filter((d) => d.kind === "audiooutput") : [];
    } catch {
      outputs = [];
    }
    if (!outputs.length) {
      const label = sinkId ? `Output ${sinkId}` : "Default system output";
      elements.device.innerHTML = `<option value="${escapeHtml(sinkId)}">${escapeHtml(label)}</option>`;
      elements.device.value = sinkId;
      return;
    }
    elements.device.innerHTML = outputs
      .map((device) => {
        const id = device.deviceId ?? "";
        return `<option value="${escapeHtml(id)}">${escapeHtml(outputLabel(device))}</option>`;
      })
      .join("");
    const selected = outputs.some((device) => (device.deviceId ?? "") === sinkId)
      ? sinkId
      : (outputs[0].deviceId ?? "");
    elements.device.value = selected;
  }

  async function selectOutputDevice() {
    const context = Tone.getContext()?.rawContext;
    if (elements.device && context && typeof context.setSinkId === "function") {
      await context.setSinkId(elements.device.value);
    }
    await refreshOutputDevice();
  }

  if (elements.device) {
    elements.device.onchange = () => selectOutputDevice();
  }

  function releasePianoNote(note, time) {
    if (!piano || !sounding.has(note.event_id)) {
      return;
    }
    sounding.delete(note.event_id);
    piano.playNoteUp({ event_id: note.event_id, pitch: note.pitch, time: time ?? Tone.now() });
  }

  function releaseAllPiano() {
    for (const note of [...sounding.values()]) {
      releasePianoNote(note);
    }
    sounding.clear();
  }

  function silenceLocally() {
    clearScheduled();
    releaseAllPiano();
    stopState(pedal);
    pedal = createPedalState();
    queued = [];
    clock = null;
  }

  function armPlayingIndicator() {
    if (!jobId || !clock) {
      return;
    }
    const start = clock.audioAt(0);
    if (Tone.now() >= start) {
      setLiveState("playing", "Playing");
      setStatus(`Live playing · job ${jobId}`);
      return;
    }
    setLiveState("buffering", `Buffering ${Math.max(0, start - Tone.now()).toFixed(1)}s`);
    scheduleAt(start, () => {
      if (jobId) {
        setLiveState("playing", "Playing");
        setStatus(`Live playing · job ${jobId}`);
      }
    });
  }

  function scheduleNoteItem(item) {
    const scheduled = scheduleNote(item.event, Tone.now(), clock);
    item.scheduled = scheduled;
    roll?.addNote(scheduled);
    if (scheduled.underrun && !item.started) {
      underruns += 1;
    }
    const velocity = clampMidiVelocity(scheduled.velocity);
    if (!item.started) {
      scheduleAt(scheduled.startAudio, (time) => {
        item.started = true;
        piano?.playNoteDown({
          event_id: scheduled.event_id,
          pitch: scheduled.pitch,
          velocity,
          time,
        });
        sounding.set(scheduled.event_id, scheduled);
        pedal.noteOn(scheduled);
      });
    }
    if (!item.ended) {
      const endAudio = item.started ? Math.max(Tone.now(), clock.audioAt(item.event.end_sec)) : scheduled.endAudio;
      scheduleAt(endAudio, (time) => {
        item.ended = true;
        const result = pedal.noteOff(item.scheduled);
        if (!result.deferred) {
          releasePianoNote(item.scheduled, time);
        }
      });
    }
  }

  function schedulePedalItem(item) {
    const scheduled = schedulePedal(item.event, Tone.now(), clock);
    item.scheduled = scheduled;
    if (scheduled.underrun && !item.fired) {
      underruns += 1;
    }
    if (!item.fired) {
      if (typeof roll?.addPedal === "function") {
        roll.addPedal(scheduled);
      }
      scheduleAt(scheduled.timeAudio, (time) => {
        item.fired = true;
        const flushed = pedal.firePedal(scheduled.value);
        for (const note of flushed) {
          releasePianoNote(note, time);
        }
      });
    }
  }

  function scheduleBeatItem(item) {
    const scheduled = schedulePedal(item.event, Tone.now(), clock);
    item.scheduled = {
      ...scheduled,
      beatKind: scheduled.beat_kind,
    };
    roll?.addBeat?.(item.scheduled);
  }

  function rescheduleQueued() {
    clearScheduled();
    armPlayingIndicator();
    for (const item of queued) {
      if (item.kind === "note" && !item.ended) {
        scheduleNoteItem(item);
      } else if (item.kind === "pedal" && !item.fired) {
        schedulePedalItem(item);
      } else if (item.kind === "beat") {
        scheduleBeatItem(item);
      }
    }
  }

  function handle(event) {
    if (event.name === "meta") {
      jobId = event.data.job_id;
      const bufferSec = Number(event.data.live_buffer_sec ?? 2);
      clock = createPlaybackClock({
        epoch: Tone.now() + bufferSec,
        speed: currentSpeed(),
      });
      queued = [];
      setLiveState("buffering", `Buffering ${bufferSec.toFixed(1)}s`);
      setStatus(`Buffering ${bufferSec.toFixed(1)}s · job ${jobId}`);
      armPlayingIndicator();
      return;
    }
    if (event.name === "midi" && event.data.kind === "note" && piano) {
      const item = { kind: "note", event: event.data, started: false, ended: false };
      queued.push(item);
      scheduleNoteItem(item);
      return;
    }
    if (event.name === "midi" && event.data.kind === "pedal") {
      const item = { kind: "pedal", event: event.data, fired: false };
      queued.push(item);
      schedulePedalItem(item);
      return;
    }
    if (event.name === "midi" && event.data.kind === "beat") {
      const item = { kind: "beat", event: event.data };
      queued.push(item);
      scheduleBeatItem(item);
      return;
    }
    if (event.name === "status") {
      const rtf = event.data.rtf == null ? "—" : Number(event.data.rtf).toFixed(2);
      setStatus(
        `tokens ${event.data.tokens ?? 0} · music ${Number(event.data.music_sec ?? 0).toFixed(1)}s · RTF ${rtf} · underruns ${underruns}`,
      );
      return;
    }
    if (event.name === "end") {
      setLiveState("idle", "Idle");
      if (event.data.reason === "error") {
        setStatus(event.data.message || "live error", "err");
      } else {
        setStatus("Live stopped", "ok");
      }
    }
  }

  async function play() {
    stop();
    cancelled = false;
    setLiveState("loading", "Loading piano");
    setStatus("Loading piano…");
    await Tone.start();
    await refreshOutputDevice();
    if (!piano) {
      piano = await loadPiano();
    }
    if (cancelled) {
      return;
    }
    silenceLocally();
    underruns = 0;
    roll?.clear();
    roll?.start();
    setLiveState("loading", "Starting generation");
    setStatus("Starting generation…");
    controller = new AbortController();
    const request = { ...getRequest(), temperature: 0.95, top_p: 0.98 };
    const response = await fetchImpl("/api/live", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: controller.signal,
    });
    if (cancelled) {
      return;
    }
    const parser = createSseParser();
    const reader = response.body.getReader();
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        for (const event of parser.push(value)) {
          handle(event);
        }
      }
    } catch (err) {
      if (err && err.name !== "AbortError") {
        setLiveState("idle", "Idle");
        setStatus(String(err), "err");
      }
    }
  }

  function stop() {
    cancelled = true;
    const currentJob = jobId;
    jobId = null;
    if (currentJob) {
      void fetchImpl(`/api/live/${currentJob}/stop`, { method: "POST" }).catch(() => {});
    }
    if (controller) {
      controller.abort();
      controller = null;
    }
    silenceLocally();
    roll?.stop();
    roll?.clear();
    setLiveState("idle", "Idle");
  }

  function syncMeta(meta) {
    if (elements.mode && Array.isArray(meta.modes)) {
      elements.mode.innerHTML = meta.modes
        .map((mode) => `<option value="${mode}">${mode}</option>`)
        .join("");
    }
    setControlsEnabled(meta.backend === "mlx");
    void refreshOutputDevice();
  }

  return { play, stop, syncMeta };
}
