
from modules.feature_extractor import extract_midi_features

print("开始读取 MIDI")

midi_file = (
    "data/maestro/maestro-v3.0.0/2004/"
    "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi"
)

features = extract_midi_features(midi_file)

print(features)
