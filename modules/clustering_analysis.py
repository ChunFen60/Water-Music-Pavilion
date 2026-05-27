import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# =========================
# 自动定位项目根目录
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# features.csv路径
features_path = os.path.join(
    BASE_DIR,
    "database",
    "features.csv"
)

# =========================
# 读取数据
# =========================
df = pd.read_csv(features_path)

print("数据读取成功")
print(df.head())

# =========================
# 聚类分析
# =========================
X = df[[
    "avg_pitch",
    "avg_duration",
    "note_count"
]]

# KMeans聚类
kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

df["cluster"] = kmeans.fit_predict(X)

print("\n聚类完成")
print(df[[
    "avg_pitch",
    "avg_duration",
    "note_count",
    "cluster"
]].head())

# =========================
# 可视化
# =========================
plt.figure(figsize=(8, 6))

plt.scatter(
    df["avg_pitch"],
    df["note_count"],
    c=df["cluster"]
)

plt.xlabel("Average Pitch")
plt.ylabel("Note Count")
plt.title("Music Style Clustering")

plt.tight_layout()
plt.show()

