import os
import pandas as pd

from midi_analyzer import analyze_midi


# =========================
# 路径
# =========================
BASE_DIR = "data/maestro/maestro-v3.0.0"

METADATA_PATH = os.path.join(
    BASE_DIR,
    "maestro-v3.0.0.csv"
)

# =========================
# 读取 metadata
# =========================
meta_df = pd.read_csv(
    METADATA_PATH
)

# =========================
# 保存结果
# =========================
results = []

# =========================
# 遍历 metadata
# =========================
for _, row in meta_df.iterrows():

    try:

        midi_rel_path = row["midi_filename"]

        midi_path = os.path.join(
            BASE_DIR,
            midi_rel_path
        )

        composer = row["canonical_composer"]

        title = row["canonical_title"]

        print(f"Analyzing: {title}")

        # MIDI 分析
        result = analyze_midi(
            midi_path
        )

        # 错误跳过
        if "error" in result:
            continue

        # 添加 metadata
        result["composer"] = composer

        result["title"] = title

        result["midi_path"] = midi_rel_path

        results.append(result)

    except Exception as e:

        print()

        print("ERROR")

        print(e)

# =========================
# DataFrame
# =========================
df = pd.DataFrame(results)

# =========================
# 删除 notes_data
# =========================
if "notes_data" in df.columns:

    df = df.drop(
        columns=["notes_data"]
    )

# =========================
# 保存 CSV
# =========================
save_path = (
    "database/music_dataset.csv"
)

df.to_csv(
    save_path,
    index=False
)

print()

print("✅ Dataset generated.")

print()

print(df.head())


