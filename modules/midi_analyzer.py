import pretty_midi
import numpy as np
from modules.key_detection import detect_key


def analyze_midi(midi_file):

    try:

        midi_data = pretty_midi.PrettyMIDI(
            midi_file
        )

    except Exception as e:

        return {
            "error": f"MIDI Parse Error: {e}"
        }

    pitches = []
    durations = []
    velocities = []
    start_times = []

    notes_data = []

    # =========================
    # 提取音符
    # =========================
    for instrument in midi_data.instruments:

        # 跳过鼓轨
        if instrument.is_drum:
            continue

        for note in instrument.notes:

            pitch = note.pitch

            duration = note.end - note.start

            velocity = note.velocity

            start_time = note.start

            pitches.append(pitch)

            durations.append(duration)

            velocities.append(velocity)

            start_times.append(start_time)

            notes_data.append({
                "pitch": pitch,
                "start": round(note.start, 3),
                "end": round(note.end, 3),
                "duration": round(duration, 3),
                "velocity": velocity
            })

    # =========================
    # 空 MIDI 检测
    # =========================
    if len(pitches) == 0:

        return {
            "error": "No notes found in MIDI file"
        }

    # =========================
    # 转 numpy
    # =========================
    pitches = np.array(pitches)

    durations = np.array(durations)

    velocities = np.array(velocities)

    start_times = np.array(start_times)

    # =========================
    # 基础特征
    # =========================
    avg_pitch = np.mean(pitches)

    avg_duration = np.mean(durations)

    avg_velocity = np.mean(velocities)

    total_notes = len(pitches)

    # =========================
    # 高级特征
    # =========================

    # 音高方差
    pitch_variance = np.var(pitches)

    # 力度方差
    velocity_variance = np.var(velocities)

    # 音域跨度
    pitch_range = np.max(pitches) - np.min(pitches)

    # MIDI 总时长
    total_time = midi_data.get_end_time()

    if total_time <= 0:
        total_time = 1

    # 音符密度
    note_density = total_notes / total_time

    # BPM
    try:
        tempo = midi_data.estimate_tempo()

    except:
        tempo = 120

    # =========================
    # 节奏复杂度
    # =========================
    rhythm_std = np.std(durations)

    # =========================
    # 音高跳跃复杂度
    # =========================
    pitch_diff = np.diff(pitches)

    melodic_complexity = np.mean(
        np.abs(pitch_diff)
    )

    # =========================
    # 情绪分析
    # =========================
    emotion = detect_emotion(
        avg_pitch,
        avg_duration,
        avg_velocity,
        note_density,
        pitch_variance,
        tempo
    )
    key_result = detect_key(midi_file)

    # =========================
    # 特征向量（机器学习）
    # =========================
    feature_vector = {

        "avg_pitch": round(avg_pitch, 2),

        "avg_duration": round(avg_duration, 2),

        "avg_velocity": round(avg_velocity, 2),

        "pitch_variance": round(
            pitch_variance,
            2
        ),

        "velocity_variance": round(
            velocity_variance,
            2
        ),

        "pitch_range": round(
            pitch_range,
            2
        ),

        "note_density": round(
            note_density,
            2
        ),

        "tempo": round(
            tempo,
            2
        ),

        "rhythm_std": round(
            rhythm_std,
            2
        ),

        "melodic_complexity": round(
            melodic_complexity,
            2
        ),

        "total_notes": total_notes
    }

    # =========================
    # 返回结果
    # =========================
    return {

        # 基础特征
        "avg_pitch": round(avg_pitch, 2),

        "avg_duration": round(avg_duration, 2),

        "avg_velocity": round(avg_velocity, 2),

        "total_notes": total_notes,

        # 高级特征
        "pitch_variance": round(
            pitch_variance,
            2
        ),

        "velocity_variance": round(
            velocity_variance,
            2
        ),

        "pitch_range": round(
            pitch_range,
            2
        ),

        "note_density": round(
            note_density,
            2
        ),

        "tempo": round(
            tempo,
            2
        ),

        "rhythm_std": round(
            rhythm_std,
            2
        ),

        "melodic_complexity": round(
            melodic_complexity,
            2
        ),

        # 情绪
        "emotion": emotion,

        # Piano Roll
        "notes_data": notes_data,

        # ML 特征
        "feature_vector": feature_vector,

        "key": key_result["key"],
        "key_confidence": key_result["confidence"]
    }


# =========================
# 情绪识别
# =========================
def detect_emotion(
    pitch,
    duration,
    velocity,
    density,
    pitch_variance,
    tempo
):

    # 激烈
    if (
        velocity > 80
        and density > 10
        and tempo > 140
    ):
        return "Energetic"

    # 悲伤
    elif (
        duration > 0.7
        and pitch < 62
        and tempo < 100
    ):
        return "Sad"

    # 黑暗
    elif (
        pitch < 58
        and pitch_variance < 45
    ):
        return "Dark"

    # 快乐
    elif (
        pitch > 67
        and velocity > 65
        and tempo > 110
    ):
        return "Happy"

    # 平静
    else:
        return "Calm"
