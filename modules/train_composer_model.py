import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================
# 读取数据
# =========================
df = pd.read_csv(
    "database/music_dataset.csv"
)

# =========================
# 删除缺失值
# =========================
df = df.dropna()

# =========================
# 特征
# =========================
feature_cols = [
    "avg_pitch",
    "avg_duration",
    "avg_velocity",
    "pitch_variance",
    "velocity_variance",
    "pitch_range",
    "note_density",
    "tempo",
    "total_notes"
]

X = df[feature_cols]

# =========================
# 标签
# =========================
y = df["composer"]

# =========================
# 标签编码
# =========================
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# =========================
# 数据集划分
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# =========================
# 模型
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

# =========================
# 训练
# =========================
model.fit(
    X_train,
    y_train
)

# =========================
# 测试
# =========================
predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print()
print("Accuracy:", accuracy)

# =========================
# 保存模型
# =========================
joblib.dump(
    model,
    "models/composer_model.pkl"
)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

print()
print("✅ Model Saved")
