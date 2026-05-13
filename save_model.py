"""
save_model.py — Obesity Risk Prediction v7
═══════════════════════════════════════════════════════════
三組特徵集 × 3演算法 × 3抽樣策略 = 27 模型實驗

特徵集版本：
  Full  (16feat): 全部特徵
  Habit (13feat): 去掉 Gender / Height / Weight
  Slim  ( 5feat): 再去掉低相關特徵，只保留 Age / FAVC / CALC / NCP / SCC

每組產出：
  📊 熱力圖（Correlation Heatmap）
  📊 3×3 F1 Grid
  📊 最佳模型混淆矩陣

執行：py save_model.py
輸出：models/  outputs/plots/
═══════════════════════════════════════════════════════════
"""

import os, copy, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"

# ════════════════════════════════════════════════════════════
# 0. 路徑
# ════════════════════════════════════════════════════════════
DATA_PATH = "data/obesity_level.csv"
MODEL_DIR = "models"
PLOT_DIR  = "outputs/plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ════════════════════════════════════════════════════════════
# 1. 讀資料 & 目標欄位
# ════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)
print(f"✅ 資料載入：{df.shape[0]} 筆, {df.shape[1]} 欄")

TARGET_COL = "0be1dad"   # 實際欄位名稱（注意：數字0開頭）

def map_obesity(label):
    if label in ["Insufficient_Weight", "Normal_Weight"]:       return 0  # Healthy
    if label in ["Overweight_Level_I",  "Overweight_Level_II"]: return 1  # Overweight
    return 2  # Obese

df["target"] = df[TARGET_COL].apply(map_obesity)

# 移除不需要的欄位
drop_cols = [c for c in ["id", TARGET_COL] if c in df.columns]
df = df.drop(columns=drop_cols)

print(f"   類別分佈: {dict(df['target'].value_counts().sort_index())}\n")

# ════════════════════════════════════════════════════════════
# 2. Label Encoding（完整版用）
# ════════════════════════════════════════════════════════════
CAT_ALL = ["Gender", "family_history_with_overweight", "FAVC",
           "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]

encoders = {}
df_enc = df.copy()
for col in CAT_ALL:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")

# ════════════════════════════════════════════════════════════
# 3. 三組特徵集定義
# ════════════════════════════════════════════════════════════
ALL16   = ["Gender","Age","Height","Weight",
           "family_history_with_overweight","FAVC","FCVC","NCP",
           "CAEC","SMOKE","CH2O","SCC","FAF","TUE","CALC","MTRANS"]

HABIT13 = [f for f in ALL16 if f not in ["Gender","Height","Weight"]]

# Slim：從 13feat 再去掉低相關特徵，只留 5 個
# 去掉：CAEC, TUE, CH2O, family_history_with_overweight, FCVC, MTRANS, FAF, SMOKE
SLIM5   = ["Age", "FAVC", "CALC", "NCP", "SCC"]

FEATURE_SETS = {
    "Full_16feat": {
        "label":    "完整特徵 (16 feat)",
        "features": ALL16,
        "df":       df_enc,
        "cat_cols": CAT_ALL,
        "save_as":  "model_full.pkl",
        "enc_save": "encoders_full.pkl",
    },
    "Habit_13feat": {
        "label":    "生活習慣 (13 feat，無 Gender/Height/Weight)",
        "features": HABIT13,
        "df":       df_enc,
        "cat_cols": [c for c in CAT_ALL if c != "Gender"],
        "save_as":  "model_habit.pkl",
        "enc_save": "encoders_habit.pkl",
    },
    "Slim_5feat": {
        "label":    "精選生活習慣 (5 feat: Age/FAVC/CALC/NCP/SCC)",
        "features": SLIM5,
        "df":       df_enc,
        "cat_cols": ["CALC"],   # SLIM5 中唯一類別欄位
        "save_as":  "model_slim.pkl",
        "enc_save": "encoders_slim.pkl",
    },
}

# ════════════════════════════════════════════════════════════
# 4. 演算法 & 抽樣策略
# ════════════════════════════════════════════════════════════
ALGORITHMS = {
    "Logistic\nRegression": LogisticRegression(max_iter=1000, random_state=42),
    "Random\nForest":       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "XGBoost":              XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                                           subsample=0.8, colsample_bytree=0.8,
                                           use_label_encoder=False, eval_metric="mlogloss",
                                           random_state=42),
}
SAMPLINGS = {
    "Raw Data":               None,
    "SMOTE":                  SMOTE(random_state=42),
    "Random\nUnder-sampling": RandomUnderSampler(random_state=42),
}
ALGO_KEYS = list(ALGORITHMS.keys())
SAMP_KEYS = list(SAMPLINGS.keys())

CLASS_NAMES  = ["Healthy", "Overweight", "Obese"]
PALETTE      = {"Full_16feat": "#2563EB", "Habit_13feat": "#16A34A", "Slim_5feat": "#DC2626"}

# ════════════════════════════════════════════════════════════
# 5. 繪圖函式
# ════════════════════════════════════════════════════════════

def plot_heatmap(df_f, features, set_name, label):
    cols = features + ["target"]
    corr = df_f[cols].corr()
    n    = len(cols)
    fig, ax = plt.subplots(figsize=(max(8, n*0.75), max(7, n*0.7)))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.3, ax=ax, annot_kws={"size": 7.5},
                cbar_kws={"shrink": 0.8})
    ax.set_title(f"Correlation Heatmap\n{label}", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    path = f"{PLOT_DIR}/{set_name}_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"   📊 Heatmap      → {path}")

def plot_f1_grid(records, set_name, label, best_algo, best_samp):
    rdf   = pd.DataFrame(records)
    pivot = rdf.pivot(index="Sampling", columns="Algorithm", values="F1")
    pivot = pivot.reindex(index=SAMP_KEYS, columns=ALGO_KEYS)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGn",
                vmin=0.45, vmax=1.0, linewidths=0.6, linecolor="white",
                ax=ax, annot_kws={"size": 12, "weight": "bold"})

    # 標記最佳格子
    best_col = ALGO_KEYS.index(best_algo)
    best_row = SAMP_KEYS.index(best_samp)
    ax.add_patch(plt.Rectangle((best_col, best_row), 1, 1,
                 fill=False, edgecolor="red", lw=3, zorder=5))

    ax.set_title(f"F1-Score Grid (3×3)\n{label}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Algorithm", fontsize=10)
    ax.set_ylabel("Sampling Strategy", fontsize=10)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    path = f"{PLOT_DIR}/{set_name}_f1_grid.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"   📊 F1 Grid      → {path}")

def plot_confusion(cm, c_names, set_name, label, algo, samp, f1):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=c_names, yticklabels=c_names,
                linewidths=0.4, ax=ax, annot_kws={"size": 13})
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)
    title = (f"Confusion Matrix — {label}\n"
             f"Best: {algo.replace(chr(10),' ')} + {samp.replace(chr(10),' ')}  "
             f"(F1={f1:.4f})")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    plt.tight_layout()
    path = f"{PLOT_DIR}/{set_name}_confusion.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"   📊 Confusion    → {path}")

# ════════════════════════════════════════════════════════════
# 6. 主實驗迴圈
# ════════════════════════════════════════════════════════════
all_best = {}

for set_name, cfg in FEATURE_SETS.items():
    print(f"\n{'═'*62}")
    print(f"  ► {cfg['label']}")
    print(f"{'═'*62}")

    df_s  = cfg["df"]
    feats = cfg["features"]
    X     = df_s[feats].values
    y     = df_s["target"].values
    n_cls = len(np.unique(y))
    c_names = CLASS_NAMES[:n_cls]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 熱力圖
    plot_heatmap(df_s, feats, set_name, cfg["label"])

    # 3×3 實驗
    records = []
    best_f1, best_model, best_algo, best_samp, best_pred = -1, None, None, None, None

    for samp_name, sampler in SAMPLINGS.items():
        for algo_name, algo_proto in ALGORITHMS.items():
            model = copy.deepcopy(algo_proto)
            if sampler is not None:
                try:
                    Xs, ys = copy.deepcopy(sampler).fit_resample(X_tr, y_tr)
                except Exception:
                    Xs, ys = X_tr, y_tr
            else:
                Xs, ys = X_tr, y_tr

            model.fit(Xs, ys)
            y_pred = model.predict(X_te)
            f1 = f1_score(y_te, y_pred, average="weighted")
            records.append({"Algorithm": algo_name, "Sampling": samp_name, "F1": f1})
            star = " ⭐" if f1 > best_f1 else ""
            a_short = algo_name.replace("\n", " ")
            s_short = samp_name.replace("\n", " ")
            print(f"   {s_short:26s} | {a_short:22s} | F1={f1:.4f}{star}")

            if f1 > best_f1:
                best_f1, best_model  = f1, model
                best_algo, best_samp = algo_name, samp_name
                best_pred            = y_pred

    # 圖表
    plot_f1_grid(records, set_name, cfg["label"], best_algo, best_samp)
    cm = confusion_matrix(y_te, best_pred)
    plot_confusion(cm, c_names, set_name, cfg["label"], best_algo, best_samp, best_f1)

    # 儲存模型 & encoder
    joblib.dump(best_model, f"{MODEL_DIR}/{cfg['save_as']}")
    # 只存這個特徵集用到的 encoder
    sub_enc = {k: encoders[k] for k in cfg["cat_cols"] if k in encoders}
    joblib.dump(sub_enc, f"{MODEL_DIR}/{cfg['enc_save']}")
    joblib.dump(feats,   f"{MODEL_DIR}/feature_cols_{set_name}.pkl")

    a_s = best_algo.replace("\n"," ")
    s_s = best_samp.replace("\n"," ")
    print(f"\n  ✅ 最佳：{a_s} + {s_s}  F1={best_f1:.4f}")
    print(classification_report(y_te, best_pred, target_names=c_names, digits=4))

    all_best[set_name] = {
        "algorithm": a_s, "sampling": s_s,
        "f1": best_f1, "n_features": len(feats),
    }

# ════════════════════════════════════════════════════════════
# 7. 摘要
# ════════════════════════════════════════════════════════════
print(f"\n{'═'*62}")
print("  最終模型摘要")
print(f"{'═'*62}")
print(f"  {'特徵集':22s} {'特徵數':6s} {'最佳演算法':22s} {'最佳抽樣':26s} F1")
for k, v in all_best.items():
    print(f"  {k:22s} {v['n_features']:<6d} {v['algorithm']:22s} {v['sampling']:26s} {v['f1']:.4f}")
print(f"\n  models/        → {len(os.listdir(MODEL_DIR))} 個檔案")
print(f"  outputs/plots/ → {len(os.listdir(PLOT_DIR))} 張圖表")
print("\n✅ 全部完成！")
