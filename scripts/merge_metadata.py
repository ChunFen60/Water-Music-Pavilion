import pandas as pd


# 读取特征数据
features_df = pd.read_csv(
    "database/features.csv"
)

# 读取官方元数据
meta_df = pd.read_csv(
    "data/maestro/maestro-v3.0.0/maestro-v3.0.0.csv"
)

# 保留需要的列
meta_df = meta_df[
    [
        "midi_filename",
        "canonical_composer",
        "canonical_title"
    ]
]

# 提取文件名
meta_df["file_name"] = meta_df["midi_filename"].apply(
    lambda x: x.split("/")[-1]
)

# 合并
merged_df = pd.merge(
    features_df,
    meta_df,
    on="file_name",
    how="left"
)

# 保存最终数据库
merged_df.to_csv(
    "database/music_dataset.csv",
    index=False
)

print("合并完成！")
print(merged_df.head())

