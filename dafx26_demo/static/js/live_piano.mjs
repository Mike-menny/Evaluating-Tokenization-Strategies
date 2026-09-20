const PIANO_PITCHES = Array.from({ length: 88 }, (_, index) => 21 + index);
const SAMPLE_ROOT = "/vendor/sgm_plus/acoustic_grand_piano";
const FADE_IN = 0.012;
const FADE_OUT = 0.2;
const MAX_VOICES = 48;

function clampMidiVelocity(velocity) {
  return Math.max(0, Math.min(127, Number(velocity)));
}

function sampleUrls() {
  return Object.fromEntries(
    PIANO_PITCHES.map((pitch) => [String(pitch), `${SAMPLE_ROOT}/p${pitch}.mp3`]),
  );
}

export async function createSampledPiano(Tone) {
  const buffers = new Tone.ToneAudioBuffers(sampleUrls());
  await Tone.loaded();
  const reverb = new Tone.Reverb({ decay: 1.8, preDelay: 0.02, wet: 0.18 }).toDestination();
  if (reverb.ready) {
    await reverb.ready;
  }
  const voices = new Map();

  function stealOldest() {
    if (voices.size < MAX_VOICES) {
      return;
    }
    const [eventId, source] = voices.entries().next().value;
    voices.delete(eventId);
    source.stop(Tone.now());
  }

  return {
    playNoteDown({ event_id, pitch, velocity, time }) {
      const buffer = buffers.get(String(pitch));
      if (!buffer) {
        return;
      }
      stealOldest();
      const source = new Tone.ToneBufferSource({
        url: buffer,
        fadeIn: FADE_IN,
        fadeOut: FADE_OUT,
      }).connect(reverb);
      source.start(time ?? Tone.now(), 0, undefined, clampMidiVelocity(velocity) / 127);
      voices.set(event_id, source);
    },
    playNoteUp({ event_id, time }) {
      const source = voices.get(event_id);
      if (!source) {
        return;
      }
      voices.delete(event_id);
      source.stop(time ?? Tone.now());
    },
  };
}
