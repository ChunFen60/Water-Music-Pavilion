import mido
import numpy as np

MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33,
    4.38, 4.09, 2.52, 5.19,
    2.39, 3.66, 2.29, 2.88
])

MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38,
    2.60, 3.53, 2.54, 4.75,
    3.98, 2.69, 3.34, 3.17
])

KEY_NAMES_MAJOR = [
    "C Major", "Db Major", "D Major", "Eb Major",
    "E Major", "F Major", "F# Major", "G Major",
    "Ab Major", "A Major", "Bb Major", "B Major"
]

KEY_NAMES_MINOR = [
    "C Minor", "C# Minor", "D Minor", "Eb Minor",
    "E Minor", "F Minor", "F# Minor", "G Minor",
    "G# Minor", "A Minor", "Bb Minor", "B Minor"
]


def extract_pitch_class_histogram(midi_file_path):
    mid = mido.MidiFile(midi_file_path)

    histogram = np.zeros(12)
    current_time = 0.0
    active_notes = {}

    for msg in mid:
        current_time += msg.time

        if msg.type in ("note_on", "note_off") and msg.channel != 9:
            key = (msg.channel, msg.note)
            is_note_on = msg.type == "note_on" and msg.velocity > 0

            if is_note_on:
                active_notes[key] = (current_time, msg.velocity, msg.note)
            else:
                if key in active_notes:
                    start_sec, velocity, pitch = active_notes.pop(key)
                    duration = current_time - start_sec
                    pitch_class = pitch % 12
                    weight = duration * (velocity / 127)
                    histogram[pitch_class] += weight

    total = np.sum(histogram)
    if total > 0:
        histogram /= total
    return histogram


def correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0
    return np.corrcoef(a, b)[0, 1]


def detect_key(midi_file_path):
    histogram = extract_pitch_class_histogram(midi_file_path)

    major_scores = []
    minor_scores = []

    for i in range(12):
        rotated = np.roll(MAJOR_PROFILE, i)
        major_scores.append(correlation(histogram, rotated))

    for i in range(12):
        rotated = np.roll(MINOR_PROFILE, i)
        minor_scores.append(correlation(histogram, rotated))

    major_scores = np.array(major_scores)
    minor_scores = np.array(minor_scores)

    best_major_idx = int(np.argmax(major_scores))
    best_minor_idx = int(np.argmax(minor_scores))

    if major_scores[best_major_idx] > minor_scores[best_minor_idx]:
        detected_key = KEY_NAMES_MAJOR[best_major_idx]
        confidence = float(major_scores[best_major_idx])
    else:
        detected_key = KEY_NAMES_MINOR[best_minor_idx]
        confidence = float(minor_scores[best_minor_idx])

    return {
        "key": detected_key,
        "confidence": round(confidence * 100, 2),
        "pitch_class_histogram": histogram.tolist(),
        "major_scores": major_scores.tolist(),
        "minor_scores": minor_scores.tolist()
    }
