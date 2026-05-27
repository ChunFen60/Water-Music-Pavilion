import pretty_midi
import pandas as pd


def extract_midi_features(midi_path):

    midi_data = pretty_midi.PrettyMIDI(midi_path)

    notes = []

    for instrument in midi_data.instruments:

        # 只分析钢琴
        if instrument.is_drum:
            continue

        for note in instrument.notes:

            notes.append({
                "pitch": note.pitch,
                "start": note.start,
                "end": note.end,
                "duration": note.end - note.start,
                "velocity": note.velocity
            })

    df = pd.DataFrame(notes)

    # 基础统计特征
    features = {
        "note_count": int(len(df)),
        "avg_pitch": float(df["pitch"].mean()),
        "avg_duration": float(df["duration"].mean()),
        "avg_velocity": float(df["velocity"].mean()),
        "max_pitch": int(df["pitch"].max()),
        "min_pitch": int(df["pitch"].min())
    }


    return features
