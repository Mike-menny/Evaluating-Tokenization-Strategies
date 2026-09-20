import { createLivePlayer } from "./live_player.mjs";
import { beatMarkerPositions, bindPlayerVisualizer } from "./midi_visualizer.mjs";

const $ = (id) => document.getElementById(id);
const SOUND_FONT = "/vendor/sgm_plus";
const results = {};

function setStatus(text, cls = "") {
  $("status").className = "status " + cls;
  $("status").textContent = text;
}

function selectedModes() {
  return [...document.querySelectorAll("#modes input:checked")].map((el) => el.value);
}

function currentPair() {
  const [composer, genre] = $("pair").value.split("::");
  return { composer, genre };
}

function featureText(features) {
  if (!features) return "";
  return [
    features.velocity ? "velocity" : "no velocity",
    features.beat ? "beat" : "no beat",
    features.pedal ? "pedal" : "no pedal",
  ].join(" · ");
}

function cardEl(mode) {
  let el = document.getElementById("card-" + mode);
  if (el) return el;
  el = document.createElement("article");
  el.className = "card panel";
  el.id = "card-" + mode;
  el.innerHTML = `<h3>${mode}</h3>
        <div class="meta" id="meta-${mode}">waiting</div>
        <midi-player id="player-${mode}" sound-font="${SOUND_FONT}" visualizer="#viz-${mode}"></midi-player>
        <midi-visualizer type="piano-roll" id="viz-${mode}"></midi-visualizer>
        <div class="beats" id="beats-${mode}"></div>
        <button class="secondary" id="retry-${mode}" type="button">Retry this mode</button>`;
  $("cards").appendChild(el);
  $("retry-" + mode).onclick = () => generateOne(mode, true);
  bindPlayerVisualizer($("player-" + mode), $("viz-" + mode));
  return el;
}

function renderResult(mode, data) {
  cardEl(mode);
  const metaEl = $("meta-" + mode);
  if (!data.ok && !data.midi_url) {
    metaEl.textContent = data.error || "failed";
    return;
  }
  const bits = [
    featureText(data.features),
    data.elapsed_sec != null ? `${data.elapsed_sec}s` : "",
    data.n_tokens ? `${data.n_tokens} tokens` : "",
    data.fallback ? "pregenerated fallback" : "",
    data.cached ? "cached" : "",
  ].filter(Boolean);
  metaEl.textContent = bits.join(" · ");
  if (data.midi_url) {
    $("player-" + mode).src = data.midi_url + "?t=" + Date.now();
  }
  const beats = $("beats-" + mode);
  beats.innerHTML = "";
  const markers = data.beat_markers || [];
  const player = $("player-" + mode);
  const renderBeats = () => {
    beats.innerHTML = "";
    const sequenceDuration = Number(player.noteSequence?.totalTime ?? 0);
    const markerDuration = Math.max(0, ...markers.map((mark) => Number(mark.time_sec)));
    for (const mark of beatMarkerPositions(markers, Math.max(sequenceDuration, markerDuration))) {
      const span = document.createElement("span");
      span.className = `beat-blip is-${mark.kind}`;
      span.style.left = `${mark.leftPercent}%`;
      span.title = mark.kind;
      beats.appendChild(span);
    }
  };
  player.addEventListener("load", renderBeats, { once: true });
  renderBeats();
}

async function generateOne(mode, retry) {
  const { composer, genre } = currentPair();
  cardEl(mode);
  $(`meta-${mode}`).textContent = retry ? "retrying…" : "composing…";
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode,
        composer,
        genre,
        seed: Number($("seed").value),
        max_tokens: Number($("max_tokens").value),
        temperature: 0.95,
        top_p: 0.98,
      }),
    });
    const data = await res.json();
    if (res.status === 429) {
      $(`meta-${mode}`).textContent = data.error;
      return;
    }
    data.ok = data.ok !== false && res.ok;
    results[mode] = data;
    renderResult(mode, data);
  } catch (err) {
    renderResult(mode, { ok: false, error: String(err) });
  }
}

async function generateAll() {
  const modes = selectedModes();
  if (modes.length < 2) {
    setStatus("Select at least two tokenization modes", "err");
    return;
  }
  $("generate").disabled = true;
  setStatus("Generating sequentially…");
  for (const mode of modes) {
    await generateOne(mode, false);
  }
  setStatus("Comparison updated. Failed cards can be retried independently.", "ok");
  $("generate").disabled = false;
}

async function loadFallback() {
  const { composer, genre } = currentPair();
  const data = await (await fetch(`/api/fallback?composer=${composer}&genre=${genre}`)).json();
  if (!data.items || !data.items.length) {
    setStatus("No quality-checked pregenerated examples for this pair", "err");
    return;
  }
  for (const item of data.items) {
    item.ok = true;
    renderResult(item.mode, item);
  }
  setStatus("Showing labeled pregenerated examples, not a live generation.", "ok");
}

async function loadRefs() {
  const data = await (await fetch("/api/references")).json();
  const root = $("refs");
  root.innerHTML = "";
  for (const item of data.items) {
    const el = document.createElement("article");
    el.className = "card panel";
    el.innerHTML = `<h3>${item.composer} · ${item.genre}</h3>
          <div class="meta">${item.dataset} · ${item.work} · ${item.license}</div>
          <div class="meta">excerpt ${item.excerpt_start_sec}–${item.excerpt_end_sec}s · ${item.available ? "on disk" : "not on this machine"}</div>`;
    if (item.midi_url) {
      const player = document.createElement("midi-player");
      player.setAttribute("sound-font", SOUND_FONT);
      player.src = item.midi_url;
      el.appendChild(player);
    }
    root.appendChild(el);
  }
}

const livePlayer = createLivePlayer({
  Tone: window.Tone,
  fetchImpl: window.fetch.bind(window),
  elements: {
    play: $("live-play"),
    stop: $("live-stop"),
    mode: $("live-mode"),
    device: $("live-audio-device"),
    indicator: $("live-indicator"),
    indicatorText: $("live-indicator-text"),
    speed: $("live-speed"),
    speedValue: $("live-speed-value"),
    roll: $("live-piano-roll"),
  },
  getRequest: () => ({
    mode: $("live-mode").value,
    composer: currentPair().composer,
    genre: currentPair().genre,
    seed: Number($("seed").value),
  }),
  setStatus,
});

async function boot() {
  try {
    const health = await (await fetch("/health")).json();
    const meta = await (await fetch("/api/meta")).json();
    $("pair").innerHTML = meta.pairs
      .map(
        (p) =>
          `<option value="${p.composer}::${p.genre}" ${p.composer === meta.defaults.composer && p.genre === meta.defaults.genre ? "selected" : ""}>${p.composer} / ${p.genre}</option>`,
      )
      .join("");
    $("modes").innerHTML = meta.modes
      .map((mode) => {
        const checked = meta.defaults.modes.includes(mode) ? "checked" : "";
        return `<label><input type="checkbox" value="${mode}" ${checked}> ${mode}</label>`;
      })
      .join("");
    livePlayer.syncMeta(meta);
    const warn = (health.missing || []).join("; ");
    setStatus(warn ? `Ready with warnings: ${warn}` : `Ready on ${health.device}`, warn ? "err" : "ok");
    await loadRefs();
  } catch (err) {
    setStatus("Could not reach the demo server", "err");
  }
}

$("generate").onclick = generateAll;
$("fallback").onclick = loadFallback;
$("live-play").onclick = () => {
  void livePlayer.play();
};
$("live-stop").onclick = () => livePlayer.stop();
$("tab-compare").onclick = () => {
  $("compare-view").classList.remove("hidden");
  $("refs-view").classList.add("hidden");
};
$("tab-refs").onclick = () => {
  $("compare-view").classList.add("hidden");
  $("refs-view").classList.remove("hidden");
};
boot();
