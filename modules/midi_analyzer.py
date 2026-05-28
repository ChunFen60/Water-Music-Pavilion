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

    emotion = _detect_emotion(avg_pitch, avg_duration, avg_velocity, note_density, pitch_variance, tempo)
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
        "emotion": emotion,
        "notes_data": notes_data,
        "feature_vector": feature_vector,
        "key": key_result["key"],
        "key_confidence": key_result["confidence"],
    }


def _detect_emotion(pitch, duration, velocity, density, pitch_variance, tempo):
    if velocity > 80 and density > 10 and tempo > 140:
        return "Energetic"
    elif duration > 0.7 and pitch < 62 and tempo < 100:
        return "Sad"
    elif pitch < 58 and pitch_variance < 45:
        return "Dark"
    elif pitch > 67 and velocity > 65 and tempo > 110:
        return "Happy"
    else:
        return "Calm"
