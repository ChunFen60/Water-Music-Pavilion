import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("features.csv")
print(df.head())

plt.figure(figsize=(10, 6))
plt.hist(df["avg_pitch"], bins=30)
plt.title("Average Pitch Distribution")
plt.xlabel("Average Pitch")
plt.ylabel("Count")
plt.show()
