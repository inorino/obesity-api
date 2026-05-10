import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib, os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/obesity_level.csv")

def map_obesity(label):
    healthy    = ['Insufficient_Weight', '0rmal_Weight']
    overweight = ['Overweight_Level_I', 'Overweight_Level_II']
    if label in healthy:      return 0
    elif label in overweight: return 1
    else:                     return 2

df['target'] = df['0be1dad'].apply(map_obesity)
df_model = df.drop(columns=['id', '0be1dad'])

cat_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le

X = df_model.drop(columns=['target'])
y = df_model['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("訓練中...")
model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
model.fit(X_train, y_train)

joblib.dump(model,                                   "models/model.pkl")
joblib.dump(encoders,                                "models/encoders.pkl")
joblib.dump({0:"健康偏瘦", 1:"過重警訊", 2:"肥胖危險群"}, "models/label_map.pkl")
joblib.dump(X.columns.tolist(),                      "models/feature_cols.pkl")

print("✅ 全部存好了！")
print("欄位順序：", X.columns.tolist())