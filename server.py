from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="肥胖預測 API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 載入你存好的四個檔案
model        = joblib.load("models/model.pkl")
encoders     = joblib.load("models/encoders.pkl")
label_map    = joblib.load("models/label_map.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

cat_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS']

# 對應你的 16 個輸入欄位（去掉 id 和 0be1dad）
class PredictRequest(BaseModel):
    Gender: str          # "Male" / "Female"
    Age: float
    Height: float        # 公尺，例如 1.75
    Weight: float        # 公斤
    family_history_with_overweight: int  # 0 或 1
    FAVC: int   # 0 或 1
    FCVC: float          # 0–3
    NCP: float           # 1–4
    CAEC: str            # "no"/"Sometimes"/"Frequently"/"Always"
    SMOKE: int   # 0 或 1
    CH2O: float          # 1–3
    SCC: int     # 0 或 1
    FAF: float           # 0–3
    TUE: float           # 0–2
    CALC: str            # "no"/"Sometimes"/"Frequently"/"Always"
    MTRANS: str          # "Automobile"/"Motorbike"/"Bike"/"Public_Transportation"/"Walking"

@app.get("/")
def root():
    return {"status": "ok", "message": "肥胖預測 API 運作中 🟢"}

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.dict()])

    # 把文字欄位用訓練時的 encoder 轉數字
    for col in cat_cols:
        df[col] = encoders[col].transform(df[col])

    # 確保欄位順序跟訓練時一樣
    df = df[feature_cols]

    # 預測
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    return {
        "prediction": label_map[int(pred)],
        "probabilities": {
            "健康偏瘦": round(float(proba[0]) * 100, 1),
            "過重警訊": round(float(proba[1]) * 100, 1),
            "肥胖危險群": round(float(proba[2]) * 100, 1),
        }
    }