import mido
import numpy as np
from modules.key_detection import MAJOR_PROFILE, MINOR_PROFILE, KEY_NAMES_MAJOR, KEY_NAMES_MINOR
from modules.harmony_analyzer import analyze_harmony


def _parse_midi_fast(midi_file_path):
    """Single-pass MIDI parse: extracts notes, tempo, and pitch-class histogram.

    Returns: (notes_data, total_time, tempo_bpm, pitch_class_histogram)
    """
    mid = mido.MidiFile(midi_file_path)
    current_time = 0.0

    active_notes = {}  # (channel, note) -> (start_sec, velocity)
    notes_data = []
    tempos = []  # collect set_tempo events
    pc_hist = np.zeros(12, dtype=float)  # pitch-class histogram for key detection

    for msg in mid:
        current_time += msg.time

        if msg.type == "set_tempo":
            tempos.append(msg.tempo)

        if msg.type in ("note_on", "note_off") and msg.channel != 9:
            key = (msg.channel, msg.note)
            is_note_on = msg.type == "note_on" and msg.velocity > 0

            if is_note_on:
                active_notes[key] = (current_time, msg.velocity)
            else:
                if key in active_notes:
                    start_sec, velocity = active_notes.pop(key)
                    duration = current_time - start_sec
                    if duration > 0:
                        notes_data.append({
                            "pitch": msg.note,
                            "start": round(start_sec, 3),
                            "end": round(current_time, 3),
                            "duration": round(duration, 3),
                            "velocity": velocity
                        })
                        # accumulate pitch-class histogram (duration-weighted)
                        pc = msg.note % 12
                        weight = duration * (velocity / 127.0)
                        pc_hist[pc] += weight

    total_time = mid.length if mid.length else current_time

    # Normalize pitch-class histogram
    total_pc = pc_hist.sum()
    if total_pc > 0:
        pc_hist /= total_pc

    # Compute tempo
    if tempos:
        avg_mspb = sum(tempos) / len(tempos)
        tempo = round(60_000_000 / avg_mspb, 2)
    else:
        tempo = 120.0

    return notes_data, total_time, tempo, pc_hist


def _detect_key_fast(pc_hist):
    """Key detection from pre-computed pitch-class histogram (no MIDI re-read)."""
    def _corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    major_scores = np.array([_corr(pc_hist, np.roll(MAJOR_PROFILE, i)) for i in range(12)])
    minor_scores = np.array([_corr(pc_hist, np.roll(MINOR_PROFILE, i)) for i in range(12)])

    best_major = int(np.argmax(major_scores))
    best_minor = int(np.argmax(minor_scores))

    if major_scores[best_major] > minor_scores[best_minor]:
        key = KEY_NAMES_MAJOR[best_major]
        confidence = float(major_scores[best_major])
    else:
        key = KEY_NAMES_MINOR[best_minor]
        confidence = float(minor_scores[best_minor])

    return {"key": key, "confidence": round(confidence * 100, 2)}


def analyze_midi(midi_file_path, skip_harmony=False):
    try:
        notes_data, total_time, tempo, pc_hist = _parse_midi_fast(midi_file_path)
    except Exception as e:
        return {"error": f"MIDI Parse Error: {e}"}

    if not notes_data:
        return {"error": "No notes found in MIDI file"}

    pitches = np.array([n["pitch"] for n in notes_data])
    durations = np.array([n["duration"] for n in notes_data])
    velocities = np.array([n["velocity"] for n in notes_data])

    avg_pitch = np.mean(pitches)
    avg_duration = np.mean(durations)
    avg_velocity = np.mean(velocities)
    total_notes = len(pitches)

    pitch_variance = np.var(pitches)
    velocity_variance = np.var(velocities)
    pitch_range = np.max(pitches) - np.min(pitches)

    if total_time <= 0:
        total_time = 1
    note_density = total_notes / total_time

    rhythm_std = np.std(durations)
    pitch_diff = np.diff(pitches)
    melodic_complexity = float(np.mean(np.abs(pitch_diff))) if len(pitch_diff) > 0 else 0.0

    avg_polyphony = _compute_polyphony(notes_data)
    pitch_entropy = _compute_pitch_entropy(notes_data)

    key_result = _detect_key_fast(pc_hist)

    # ---- Harmony analysis (skip in batch mode for speed) ----
    if skip_harmony:
        harmony = {
            "chord_sequence": [], "chord_types": {}, "chord_diversity": 0.0,
            "harmonic_rhythm": 0.0, "cadences": [], "cadence_counts": {},
            "bass_motion_step_ratio": 0.0, "tonic_dominant_ratio": 0.5,
            "chromatic_pct": 0.0, "total_chords_detected": 0,
        }
    else:
        harmony = analyze_harmony(notes_data, key_result["key"])

    def r(v):
        return round(float(v), 2)

    feature_vector = {
        "avg_pitch": r(avg_pitch),
        "avg_duration": r(avg_duration),
        "avg_velocity": r(avg_velocity),
        "pitch_variance": r(pitch_variance),
        "velocity_variance": r(velocity_variance),
        "pitch_range": r(pitch_range),
        "note_density": r(note_density),
        "tempo": r(tempo),
        "rhythm_std": r(rhythm_std),
        "melodic_complexity": r(melodic_complexity),
        "total_notes": total_notes,
        "avg_polyphony": r(avg_polyphony),
        "pitch_entropy": r(pitch_entropy),
        # harmony summary
        "chord_diversity": harmony["chord_diversity"],
        "harmonic_rhythm": harmony["harmonic_rhythm"],
        "tonic_dominant_ratio": r(harmony["tonic_dominant_ratio"]),
        "bass_motion_step_ratio": harmony["bass_motion_step_ratio"],
        "chromatic_pct": harmony["chromatic_pct"],
    }

    return {
        "avg_pitch": r(avg_pitch),
        "avg_duration": r(avg_duration),
        "avg_velocity": r(avg_velocity),
        "total_notes": total_notes,
        "pitch_variance": r(pitch_variance),
        "velocity_variance": r(velocity_variance),
        "pitch_range": r(pitch_range),
        "note_density": r(note_density),
        "tempo": r(tempo),
        "rhythm_std": r(rhythm_std),
        "melodic_complexity": r(melodic_complexity),
        "notes_data": notes_data,
        "feature_vector": feature_vector,
        "key": key_result["key"],
        "key_confidence": key_result["confidence"],
        "avg_polyphony": r(avg_polyphony),
        "pitch_entropy": r(pitch_entropy),
        # harmony features
        "chord_types": harmony["chord_types"],
        "chord_diversity": harmony["chord_diversity"],
        "harmonic_rhythm": harmony["harmonic_rhythm"],
        "cadences": harmony["cadences"],
        "cadence_counts": harmony["cadence_counts"],
        "bass_motion_step_ratio": harmony["bass_motion_step_ratio"],
        "tonic_dominant_ratio": r(harmony["tonic_dominant_ratio"]),
        "chromatic_pct": harmony["chromatic_pct"],
        "total_chords_detected": harmony["total_chords_detected"],
    }


def _compute_polyphony(notes_data):
    """Average number of simultaneously sounding notes over time."""
    if not notes_data:
        return 0
    events = []
    for n in notes_data:
        events.append((n["start"], 1))
        events.append((n["end"], -1))
    events.sort(key=lambda x: x[0])
    current = 0
    weighted = 0.0
    prev_t = events[0][0]
    for t, delta in events:
        weighted += current * (t - prev_t)
        prev_t = t
        current += delta
    total_t = events[-1][0] - events[0][0]
    return weighted / total_t if total_t > 0 else 0


def _compute_pitch_entropy(notes_data):
    """Normalized entropy of pitch-class distribution (0-1).
    Higher = more chromatic (Romantic). Lower = more diatonic (Baroque)."""
    if not notes_data:
        return 0
    histogram = np.zeros(12)
    total_w = 0.0
    for n in notes_data:
        w = n["duration"] * (n["velocity"] / 127)
        histogram[n["pitch"] % 12] += w
        total_w += w
    if total_w > 0:
        histogram /= total_w
    entropy = -np.sum(histogram * np.log(histogram + 1e-12))
    return entropy / np.log(12)
