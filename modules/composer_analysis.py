import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("database/music_dataset.csv")
df = df.dropna(subset=["canonical_composer"])
composer_stats = df.groupby(
    "canonical_composer"
)[
    [
        "avg_pitch",
        "avg_velocity",
        "note_count"
    ]
].mean()
top_composers = df["canonical_composer"].value_counts().head(10).index
composer_stats = composer_stats.loc[top_composers]
print(composer_stats)
plt.figure(figsize=(12, 6))
composer_stats["avg_pitch"].sort_values().plot(
    kind="barh"
)
plt.title("Composer Average Pitch Comparison")
plt.xlabel("Average Pitch")
plt.ylabel("Composer")
plt.show()


plt.figure(figsize=(12, 6))
composer_stats["avg_velocity"].sort_values().plot(
    kind="barh"
)
plt.title("Composer Average Velocity Comparison")
plt.xlabel("Average Velocity")
plt.ylabel("Composer")
plt.show()
plt.figure(figsize=(12, 6))
composer_stats["note_count"].sort_values().plot(
    kind="barh"
)

plt.title("Composer Note Count Comparison")
plt.xlabel("Average Note Count")
plt.ylabel("Composer")
plt.show()

