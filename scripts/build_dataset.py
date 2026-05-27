
import os
import pandas as pd

from modules.feature_extractor import extract_midi_features


# 数据集根目录
base_path = "data/maestro/maestro-v3.0.0"

# 存储所有特征
all_features = []


# 遍历所有文件夹
for root, dirs, files in os.walk(base_path):

    for file in files:

        # 只处理 MIDI 文件
        if file.endswith(".midi"):

            midi_path = os.path.join(root, file)

            try:

                print(f"正在分析: {file}")

                # 提取特征
                features = extract_midi_features(midi_path)

                # 添加文件名
                features["file_name"] = file

                # 添加年份
                features["year"] = root.split("\\")[-1]

                # 保存
                all_features.append(features)

            except Exception as e:

                print(f"错误文件: {file}")
                print(e)


# 转为 DataFrame
df = pd.DataFrame(all_features)

# 保存 CSV
df.to_csv("features.csv", index=False)

print("\n全部完成！")
print(df.head())