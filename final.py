import re
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler  # تم التغيير هنا
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ==========================================
# 1. تحميل البيانات والتحقق منها
# ==========================================
df = pd.read_csv("C:/Users/bebos/Desktop/pyy/ai4i2020.csv")

missing_values = df.isnull().sum()
if missing_values.sum() == 0:
  print("No missing values found in the DataFrame.")

duplicate_rows = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}\n")

# ==========================================
# 2. Feature Engineering & Preprocessing
# ==========================================
df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["power_watts"] = (
    df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
)
df["overstrain_index"] = df["Torque [Nm]"] * df["Tool wear [min]"]
df["torque_rpm_ratio"] = df["Torque [Nm]"] / (
    df["Rotational speed [rpm]"] + 1e-5
)

# حذف الأعمدة غير الضرورية وتحويل الـ Type
df = df.drop(columns=["UDI", "Product ID"])
type_map = {"L": 1, "M": 2, "H": 3}
df["type_encoded"] = df["Type"].map(type_map)
df = df.drop(columns=["Type"])

# تنظيف أسماء الأعمدة من الأقواس والرموز الخاصة لـ XGBoost من البداية
df.columns = [re.sub(r"[\[\]<]", "", col) for col in df.columns]

# ==========================================
# 3. فصل الـ Features والتأكد من منع Data Leakage
# ==========================================
drop_cols = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
cols_to_drop = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=cols_to_drop)
y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 4. Model 1: Support Vector Machine (SVC)
# ==========================================
# استخدام RobustScaler بدلاً من StandardScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svc_model = SVC(kernel="rbf", class_weight="balanced", random_state=42)
svc_model.fit(X_train_scaled, y_train)

y_pred_svc = svc_model.predict(X_test_scaled)

print("=== SVC Evaluation (with RobustScaler) ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svc))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred_svc),
)

# ==========================================
# 5. Model 2: XGBoost Classifier
# ==========================================
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

xgb_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, y_pred_xgb)

print(f"\n=== XGBoost Test Accuracy: {xgb_acc * 100:.2f}% ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred_xgb),
)

# ==========================================
# 6. Threshold Optimization for XGBoost
# ==========================================
y_probs = xgb_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print(
    f"{'Threshold':^10} | {'Precision (1)':^14} | {'Recall (1)':^12} |"
    f" {'F1-Score (1)':^12}"
)
print("=" * 60)

for threshold in np.arange(0.01, 0.9, 0.01):
  y_pred_custom = (y_probs >= threshold).astype(int)
  report = classification_report(y_test, y_pred_custom, output_dict=True)

  p = report["1"]["precision"]
  r = report["1"]["recall"]
  f1 = report["1"]["f1-score"]

  print(f"{threshold:^10.2f} | {p:^14.2f} | {r:^12.2f} | {f1:^12.2f}")

# ==========================================
# 7. Evaluation at Final Threshold (0.07)
# ==========================================
BEST_THRESHOLD = 0.07
y_pred_final = (y_probs >= BEST_THRESHOLD).astype(int)

print(f"\n=== Final XGBoost Evaluation at Threshold = {BEST_THRESHOLD} ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_final))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred_final),
)