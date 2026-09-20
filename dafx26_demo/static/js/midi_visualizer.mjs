export const PIANO_ROLL_CONFIG = {
  noteRGB: "196, 163, 90",
  activeNoteRGB: "244, 238, 228",
  noteHeight: 6,
  pixelsPerTimeStep: 28,
};

export function beatMarkerPositions(markers, duration) {
  const total = Number(duration);
  if (!(total > 0)) {
    return [];
  }
  return markers.map((marker) => ({
    leftPercent: Math.max(0, Math.min(100, (Number(marker.time_sec) / total) * 100)),
    kind: marker.kind,
  }));
}

export function bindPlayerVisualizer(player, visualizer) {
  visualizer.config = { ...PIANO_ROLL_CONFIG };
  const sync = () => {
    if (player.noteSequence) {
      visualizer.noteSequence = player.noteSequence;
    }
  };
  player.addEventListener("load", sync);
  sync();
  return sync;
}
