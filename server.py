"""
server.py — Obesity Risk Prediction API v7
═══════════════════════════════════════════
端點：
  GET  /              → 服務狀態
  POST /predict       → 完整版 (16 feat)
  POST /predict_habit → 生活習慣版 (13 feat)
  POST /predict_slim  → 精選版 (5 feat: Age/FAVC/CALC/NCP/SCC)
  POST /predict_all   → 三組模型同時比較

執行：py -m uvicorn server:app --reload --port 8000
文件：http://localhost:8000/docs
═══════════════════════════════════════════
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict
import joblib, numpy as np, os

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Obesity Risk Prediction API",
    description="""
## 🏥 肥胖風險預測 API v7

三組特徵集各自跑 3×3 實驗，選出最佳模型後部署。

| 端點 | 特徵集 | 特徵數 | 說明 |
|------|--------|--------|------|
| `/predict` | Full | **16** | 完整身體 + 生活習慣，精準度最高 |
| `/predict_habit` | Habit | **13** | 無需身高體重性別，純生活習慣 |
| `/predict_slim` | Slim | **5** | 最精簡：Age / FAVC / CALC / NCP / SCC |
| `/predict_all` | All | — | 三組同時比較 |

### 預測類別
- `0` Healthy（健康/偏瘦）
- `1` Overweight（過重）
- `2` Obese（肥胖）

---
**Cloud**: https://obesity-api-mj6a.onrender.com  
**GitHub**: https://github.com/inorino/obesity-api
    """,
    version="7.0.0",
)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── 載入模型 ─────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.join(BASE, "models")

def _load(name):
    p = os.path.join(MDIR, name)
    if not os.path.exists(p):
        raise RuntimeError(f"❌ 找不到 {p}，請先執行 save_model.py")
    return joblib.load(p)

model_full  = _load("model_full.pkl")
model_habit = _load("model_habit.pkl")
model_slim  = _load("model_slim.pkl")
enc_full    = _load("encoders_full.pkl")
enc_habit   = _load("encoders_habit.pkl")
enc_slim    = _load("encoders_slim.pkl")

LABEL = {0: "Healthy", 1: "Overweight", 2: "Obese"}

# ── 編碼對照表（API 接受可讀字串） ──────────────────────────
BINARY = {"yes": 1, "no": 0}
FREQ3  = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
TRANSPORT = {"Walking": 0, "Bike": 1, "Motorbike": 2,
             "Public_Transportation": 3, "Automobile": 4}

def enc(field, val, mapping):
    if val not in mapping:
        raise HTTPException(422,
            detail=f"'{field}' 值 '{val}' 無效，允許：{list(mapping.keys())}")
    return mapping[val]

def bmi_calc(h, w):
    if h is None or w is None: return None
    if h <= 0: raise HTTPException(422, detail="Height 必須 > 0")
    return round(w / h**2, 2)

def prob_dict(model, X):
    p = model.predict_proba(X)[0]
    return {LABEL[i]: round(float(v), 4) for i, v in enumerate(p)}

def make_resp(model, X, name, bmi_val):
    pred = int(model.predict(X)[0])
    return {"prediction": pred, "label": LABEL[pred],
            "probability": prob_dict(model, X),
            "bmi": bmi_val, "model": name}

# ════════════════════════════════════════════════════════════
# Pydantic Schemas
# ════════════════════════════════════════════════════════════

class FullInput(BaseModel):
    """16 個特徵（完整版）"""
    Gender:                        str   = Field(..., example="Male",    description="Male / Female")
    Age:                           float = Field(..., example=27)
    Height:                        float = Field(..., example=1.75,      description="公尺，e.g. 1.75")
    Weight:                        float = Field(..., example=82,        description="公斤")
    family_history_with_overweight:str   = Field(..., example="yes",     description="yes / no")
    FAVC:                          str   = Field(..., example="yes",     description="常吃高熱量食物 yes/no")
    FCVC:                          float = Field(..., example=2.0,       description="蔬菜攝取頻率 0-3")
    NCP:                           float = Field(..., example=3.0,       description="每日主餐次數")
    CAEC:                          str   = Field(..., example="Sometimes",description="no/Sometimes/Frequently/Always")
    SMOKE:                         str   = Field(..., example="no",      description="yes / no")
    CH2O:                          float = Field(..., example=2.0,       description="每日飲水公升")
    SCC:                           str   = Field(..., example="no",      description="計算卡路里 yes/no")
    FAF:                           float = Field(..., example=0.5,       description="每週運動頻率 0-3")
    TUE:                           float = Field(..., example=1.0,       description="每日3C時數 0-2")
    CALC:                          str   = Field(..., example="Sometimes",description="飲酒 no/Sometimes/Frequently/Always")
    MTRANS:                        str   = Field(..., example="Public_Transportation",
                                                    description="Walking/Bike/Motorbike/Public_Transportation/Automobile")

class HabitInput(BaseModel):
    """13 個特徵（生活習慣版，無 Gender/Height/Weight）"""
    Age:                           float = Field(..., example=27)
    Height:                        Optional[float] = Field(None, example=1.75, description="選填，有填才算 BMI")
    Weight:                        Optional[float] = Field(None, example=82,   description="選填，有填才算 BMI")
    family_history_with_overweight:str   = Field(..., example="yes")
    FAVC:                          str   = Field(..., example="yes")
    FCVC:                          float = Field(..., example=2.0)
    NCP:                           float = Field(..., example=3.0)
    CAEC:                          str   = Field(..., example="Sometimes")
    SMOKE:                         str   = Field(..., example="no")
    CH2O:                          float = Field(..., example=2.0)
    SCC:                           str   = Field(..., example="no")
    FAF:                           float = Field(..., example=0.5)
    TUE:                           float = Field(..., example=1.0)
    CALC:                          str   = Field(..., example="Sometimes")
    MTRANS:                        str   = Field(..., example="Public_Transportation")

class SlimInput(BaseModel):
    """5 個特徵（精選版）：Age / FAVC / CALC / NCP / SCC"""
    Age:   float = Field(..., example=27,          description="年齡")
    FAVC:  str   = Field(..., example="yes",       description="常吃高熱量食物 yes/no")
    CALC:  str   = Field(..., example="Sometimes", description="飲酒頻率 no/Sometimes/Frequently/Always")
    NCP:   float = Field(..., example=3.0,         description="每日主餐次數")
    SCC:   str   = Field(..., example="no",        description="是否計算卡路里 yes/no")
    # 選填，只用來回傳 BMI
    Height:Optional[float] = Field(None, example=1.75, description="選填，有填才算 BMI")
    Weight:Optional[float] = Field(None, example=82,   description="選填，有填才算 BMI")

# ════════════════════════════════════════════════════════════
# 特徵向量建構
# ════════════════════════════════════════════════════════════

def vec_full(d: FullInput):
    return np.array([[
        enc("Gender",  d.Gender,  {"Male":1,"Female":0}),
        d.Age, d.Height, d.Weight,
        enc("family_history_with_overweight", d.family_history_with_overweight, BINARY),
        enc("FAVC",  d.FAVC,  BINARY),
        d.FCVC, d.NCP,
        enc("CAEC",  d.CAEC,  FREQ3),
        enc("SMOKE", d.SMOKE, BINARY),
        d.CH2O,
        enc("SCC",   d.SCC,   BINARY),
        d.FAF, d.TUE,
        enc("CALC",  d.CALC,  FREQ3),
        enc("MTRANS",d.MTRANS,TRANSPORT),
    ]])

def vec_habit(d: HabitInput):
    return np.array([[
        d.Age,
        enc("family_history_with_overweight", d.family_history_with_overweight, BINARY),
        enc("FAVC",  d.FAVC,  BINARY),
        d.FCVC, d.NCP,
        enc("CAEC",  d.CAEC,  FREQ3),
        enc("SMOKE", d.SMOKE, BINARY),
        d.CH2O,
        enc("SCC",   d.SCC,   BINARY),
        d.FAF, d.TUE,
        enc("CALC",  d.CALC,  FREQ3),
        enc("MTRANS",d.MTRANS,TRANSPORT),
    ]])

def vec_slim(d: SlimInput):
    return np.array([[
        d.Age,
        enc("FAVC", d.FAVC, BINARY),
        enc("CALC", d.CALC, FREQ3),
        d.NCP,
        enc("SCC",  d.SCC,  BINARY),
    ]])

# ════════════════════════════════════════════════════════════
# 端點
# ════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"], summary="服務狀態")
def health():
    return {
        "status": "ok", "version": "7.0.0",
        "endpoints": ["/predict", "/predict_habit", "/predict_slim", "/predict_all"],
        "docs": "/docs", "redoc": "/redoc",
    }

@app.post("/predict", tags=["Predict"], summary="完整版（16 features）")
def predict(data: FullInput):
    """
    最高精準度。使用 16 個完整特徵（身體數據 + 生活習慣）。
    自動計算 BMI（Height / Weight 為必填）。
    """
    X = vec_full(data)
    b = bmi_calc(data.Height, data.Weight)
    return make_resp(model_full, X, "Best_Full_16feat", b)

@app.post("/predict_habit", tags=["Predict"], summary="生活習慣版（13 features）")
def predict_habit(data: HabitInput):
    """
    不需要 Gender / Height / Weight，純靠生活習慣預測。
    Height & Weight 為選填（有填才回傳 BMI，否則 bmi=null）。
    """
    X = vec_habit(data)
    b = bmi_calc(data.Height, data.Weight)
    return make_resp(model_habit, X, "Best_Habit_13feat", b)

@app.post("/predict_slim", tags=["Predict"], summary="精選版（5 features）")
def predict_slim(data: SlimInput):
    """
    最精簡輸入，只需要 5 個欄位：
    **Age / FAVC / CALC / NCP / SCC**

    Height & Weight 選填，有填才計算 BMI。
    """
    X = vec_slim(data)
    b = bmi_calc(data.Height, data.Weight)
    return make_resp(model_slim, X, "Best_Slim_5feat", b)

@app.post("/predict_all", tags=["Predict"], summary="ALL IN ONE — 三組模型同時比較")
def predict_all(data: FullInput):
    """
    同一份輸入，一次跑三個最佳模型，回傳三組預測結果。

    - **full_16feat** → 完整版 16 feat
    - **habit_13feat** → 生活習慣版 13 feat（自動忽略 Gender/Height/Weight）
    - **slim_5feat** → 精選版 5 feat（只用 Age/FAVC/CALC/NCP/SCC）
    """
    b = bmi_calc(data.Height, data.Weight)

    habit_d = HabitInput(
        Age=data.Age, Height=data.Height, Weight=data.Weight,
        family_history_with_overweight=data.family_history_with_overweight,
        FAVC=data.FAVC, FCVC=data.FCVC, NCP=data.NCP, CAEC=data.CAEC,
        SMOKE=data.SMOKE, CH2O=data.CH2O, SCC=data.SCC,
        FAF=data.FAF, TUE=data.TUE, CALC=data.CALC, MTRANS=data.MTRANS,
    )
    slim_d = SlimInput(
        Age=data.Age, FAVC=data.FAVC, CALC=data.CALC,
        NCP=data.NCP, SCC=data.SCC,
        Height=data.Height, Weight=data.Weight,
    )

    return {
        "full_16feat":  make_resp(model_full,  vec_full(data),     "Best_Full_16feat",  b),
        "habit_13feat": make_resp(model_habit, vec_habit(habit_d), "Best_Habit_13feat", b),
        "slim_5feat":   make_resp(model_slim,  vec_slim(slim_d),   "Best_Slim_5feat",   b),
        "bmi": b,
    }
