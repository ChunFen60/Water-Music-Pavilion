
import pretty_midi
import pandas as pd
import matplotlib.pyplot as plt
import random

# =========================
# 读取 MAESTRO metadata
# =========================
csv_path = "data/maestro/maestro-v3.0.0/maestro-v3.0.0.csv"

df = pd.read_csv(csv_path)

print("数据集读取成功")
print("总曲目数量：", len(df))

# =========================
# 随机选择一首曲子
# =========================
sample = df.sample(1).iloc[0]

midi_relative_path = sample["midi_filename"]

# 拼接完整路径
midi_path = "data/maestro/maestro-v3.0.0/" + midi_relative_path

print("\n当前分析曲目：")
print("作曲家：", sample["canonical_composer"])
print("作品：", sample["canonical_title"])

print("\nMIDI路径：")
print(midi_path)

# =========================
# 读取 MIDI
# =========================
print("\n开始读取 MIDI...")

midi_data = pretty_midi.PrettyMIDI(midi_path)

print("MIDI读取成功")

# =========================
# 生成 Piano Roll
# =========================
print("生成 Piano Roll...")

piano_roll = midi_data.get_piano_roll(fs=100)

# =========================
# 绘图
# =========================
plt.figure(figsize=(14, 6))

plt.imshow(
    piano_roll,
    aspect='auto',
    origin='lower'
)

plt.title(
    f"Piano Roll\n{sample['canonical_composer']} - {sample['canonical_title']}"
)

plt.xlabel("Time")
plt.ylabel("Pitch")

plt.colorbar(label="Velocity")

plt.tight_layout()

plt.show()

print("绘制完成")



