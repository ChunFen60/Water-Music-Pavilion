
import os
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# features.csv路径
features_path = os.path.join(
    BASE_DIR,
    "database",
    "features.csv"
)


df = pd.read_csv(features_path)

print("数据读取成功")
print(df.head())

def classify_emotion(row):

    avg_pitch = row["avg_pitch"]
    note_count = row["note_count"]
    avg_duration = row["avg_duration"]

    if avg_pitch > 66 and note_count > 5000:
        return "Passionate"

    elif avg_duration > 0.5:
        return "Gentle"

    elif avg_pitch < 60:
        return "Sad"

    else:
        return "Neutral"

# 添加情绪列
df["emotion"] = df.apply(classify_emotion, axis=1)

print("\n情绪分类完成")
print(df[["emotion"]].head())


emotion_counts = df["emotion"].value_counts()

print("\n情绪统计：")
print(emotion_counts)


plt.figure(figsize=(8, 5))

emotion_counts.plot(kind="bar")

plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
 
