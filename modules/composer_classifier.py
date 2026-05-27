import pandas as pd
import numpy as np


def classify_composer(
    input_features,
    dataset_path
):

    # 读取数据
    df = pd.read_csv(dataset_path)

    # 必须字段
    required_cols = [
        "canonical_composer",
        "avg_pitch",
        "avg_duration"
    ]

    for col in required_cols:

        if col not in df.columns:

            return {
                "error": f"Missing column: {col}"
            }

    # 去掉空值
    df = df.dropna(
        subset=[
            "avg_pitch",
            "avg_duration"
        ]
    )

    # 计算距离
    distances = []

    for _, row in df.iterrows():

        distance = np.sqrt(

            (row["avg_pitch"]
             - input_features["avg_pitch"]) ** 2

            +

            (row["avg_duration"]
             - input_features["avg_duration"]) ** 2
        )

        distances.append({
            "composer":
                row["canonical_composer"],

            "distance":
                distance
        })

    # 排序
    distances = sorted(
        distances,
        key=lambda x: x["distance"]
    )

    # 最近作曲家
    best_match = distances[0]

    # 相似度
    similarity = max(
        0,
        100 - best_match["distance"] * 10
    )

    return {

        "composer":
            best_match["composer"],

        "similarity":
            round(similarity, 2)
    }
