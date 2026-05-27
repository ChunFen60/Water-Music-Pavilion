import numpy as np
import pretty_midi

#MIR调性分析模型
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
    "C Major",
    "Db Major",
    "D Major",
    "Eb Major",
    "E Major",
    "F Major",
    "F# Major",
    "G Major",
    "Ab Major",
    "A Major",
    "Bb Major",
    "B Major"
]

KEY_NAMES_MINOR = [
    "C Minor",
    "C# Minor",
    "D Minor",
    "Eb Minor",
    "E Minor",
    "F Minor",
    "F# Minor",
    "G Minor",
    "G# Minor",
    "A Minor",
    "Bb Minor",
    "B Minor"
]



def extract_pitch_class_histogram(midi_data):

    histogram = np.zeros(12)

    for instrument in midi_data.instruments:

        if instrument.is_drum:
            continue

        for note in instrument.notes:

            pitch_class = note.pitch % 12

            duration = note.end - note.start

            velocity_weight = note.velocity / 127

            weight = duration * velocity_weight

            histogram[pitch_class] += weight

    # 归一化
    total = np.sum(histogram)

    if total > 0:
        histogram /= total

    return histogram


# =========================
# 计算相关性
# =========================

def correlation(a, b):

    if np.std(a) == 0 or np.std(b) == 0:
        return 0

    return np.corrcoef(a, b)[0, 1]


# =========================
# 调性检测
# =========================

def detect_key(midi_file):

    midi_data = pretty_midi.PrettyMIDI(midi_file)

    histogram = extract_pitch_class_histogram(
        midi_data
    )

    major_scores = []
    minor_scores = []

    # =========================
    # 所有大调
    # =========================

    for i in range(12):

        rotated_profile = np.roll(
            MAJOR_PROFILE,
            i
        )

        score = correlation(
            histogram,
            rotated_profile
        )

        major_scores.append(score)

    # =========================
    # 所有小调
    # =========================

    for i in range(12):

        rotated_profile = np.roll(
            MINOR_PROFILE,
            i
        )

        score = correlation(
            histogram,
            rotated_profile
        )

        minor_scores.append(score)

    major_scores = np.array(major_scores)
    minor_scores = np.array(minor_scores)

    best_major_idx = np.argmax(major_scores)
    best_minor_idx = np.argmax(minor_scores)

    best_major_score = major_scores[
        best_major_idx
    ]

    best_minor_score = minor_scores[
        best_minor_idx
    ]

    # =========================
    # 判断最终调性
    # =========================

    if best_major_score > best_minor_score:

        detected_key = KEY_NAMES_MAJOR[
            best_major_idx
        ]

        confidence = best_major_score

    else:

        detected_key = KEY_NAMES_MINOR[
            best_minor_idx
        ]

        confidence = best_minor_score

    return {
        "key": detected_key,
        "confidence": round(
            confidence * 100,
            2
        ),
        "pitch_class_histogram": histogram.tolist(),
        "major_scores": major_scores.tolist(),
        "minor_scores": minor_scores.tolist()
    }
