import mido
import numpy as np
from modules.key_detection import detect_key


def _parse_midi(midi_file_path):
    mid = mido.MidiFile(midi_file_path)
    current_time = 0.0

    active_notes = {}  # (channel, note) -> (start_sec, velocity)
    notes_data = []

    for msg in mid:
        current_time += msg.time

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

    total_time = mid.length if mid.length else current_time
    return notes_data, total_time


def _estimate_tempo(midi_file_path):
    mid = mido.MidiFile(midi_file_path)
    tempos = []
    for msg in mid:
        if msg.type == "set_tempo":
            tempos.append(msg.tempo)
    if tempos:
        avg_mspb = sum(tempos) / len(tempos)
        return round(60_000_000 / avg_mspb, 2)
    return 120  # default


def analyze_midi(midi_file_path):
    try:
        notes_data, total_time = _parse_midi(midi_file_path)
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

    try:
        tempo = _estimate_tempo(midi_file_path)
    except Exception:
        tempo = 120

    rhythm_std = np.std(durations)
    pitch_diff = np.diff(pitches)
    melodic_complexity = float(np.mean(np.abs(pitch_diff))) if len(pitch_diff) > 0 else 0.0

    avg_polyphony = _compute_polyphony(notes_data)
    pitch_entropy = _compute_pitch_entropy(notes_data)

    key_result = detect_key(midi_file_path)

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
