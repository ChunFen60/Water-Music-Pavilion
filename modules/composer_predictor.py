import joblib
import numpy as np
import os

# =========================
# 加载模型
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "composer_model.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "label_encoder.pkl"
)

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(
    ENCODER_PATH
)

# =========================
# 作曲家预测
# =========================
def predict_composer(result):

    features = np.array([[
        result["avg_pitch"],
        result["avg_duration"],
        result["avg_velocity"],
        result["pitch_variance"],
        result["velocity_variance"],
        result["pitch_range"],
        result["note_density"],
        result["tempo"]
    ]])

    prediction = model.predict(features)

    composer = label_encoder.inverse_transform(
        prediction
    )[0]

    return composer