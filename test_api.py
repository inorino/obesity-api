"""
test_api.py — 測試所有端點 v7
本機：py test_api.py
雲端：py test_api.py --cloud
"""
import sys, json, requests

BASE = "https://obesity-api-mj6a.onrender.com" if "--cloud" in sys.argv \
       else "http://localhost:8000"
print(f"\n🔗 目標：{BASE}")
print("（第一次雲端呼叫可能需等 30 秒 Render 喚醒）")
print("=" * 60)

# ── Payload ──────────────────────────────────────────────────
FULL_PAYLOAD = {
    "Gender": "Male", "Age": 27.0, "Height": 1.75, "Weight": 82.0,
    "family_history_with_overweight": "yes",
    "FAVC": "yes", "FCVC": 2.0, "NCP": 3.0,
    "CAEC": "Sometimes", "SMOKE": "no",
    "CH2O": 2.0, "SCC": "no",
    "FAF": 0.5, "TUE": 1.0,
    "CALC": "Sometimes",
    "MTRANS": "Public_Transportation",
}

# Habit: 去掉 Gender，Height/Weight 選填
HABIT_PAYLOAD = {
    "Age": 27.0, "Height": 1.75, "Weight": 82.0,
    "family_history_with_overweight": "yes",
    "FAVC": "yes", "FCVC": 2.0, "NCP": 3.0,
    "CAEC": "Sometimes", "SMOKE": "no",
    "CH2O": 2.0, "SCC": "no",
    "FAF": 0.5, "TUE": 1.0,
    "CALC": "Sometimes",
    "MTRANS": "Public_Transportation",
}

# Slim: 只有 5 個欄位 + 選填 Height/Weight
SLIM_PAYLOAD = {
    "Age": 27.0, "FAVC": "yes",
    "CALC": "Sometimes", "NCP": 3.0, "SCC": "no",
    # 選填 BMI
    "Height": 1.75, "Weight": 82.0,
}

# 高 BMI 個案測試
HIGH_BMI_FULL = dict(FULL_PAYLOAD)
HIGH_BMI_FULL.update({"Height": 1.65, "Weight": 100.0, "Age": 40.0, "FAF": 0.0})

def test(method, path, payload=None, label=""):
    url = f"{BASE}{path}"
    try:
        r = (requests.get(url, timeout=35) if method == "GET"
             else requests.post(url, json=payload, timeout=35))
        ok = "✅" if r.status_code == 200 else "❌"
        print(f"\n{ok} [{r.status_code}]  {label or path}")
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌  {path}  錯誤：{e}")

# ── 測試 ─────────────────────────────────────────────────────
test("GET",  "/",               label="Health Check")
test("POST", "/predict",        FULL_PAYLOAD,   "完整版 Full  (16 feat)")
test("POST", "/predict_habit",  HABIT_PAYLOAD,  "習慣版 Habit (13 feat)")
test("POST", "/predict_slim",   SLIM_PAYLOAD,   "精選版 Slim  ( 5 feat): Age/FAVC/CALC/NCP/SCC")
test("POST", "/predict_all",    FULL_PAYLOAD,   "ALL IN ONE（三組同時）")
test("POST", "/predict_all",    HIGH_BMI_FULL,  "加碼：高 BMI 個案（身高 1.65 / 體重 100kg）")

print(f"\n{'='*60}")
print(f"✅ 測試完成！互動文件：{BASE}/docs\n")
