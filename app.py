import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load
df = pd.read_csv("emotions.csv")
df = df.dropna(subset=["label"])
print("✅ Loaded:", df.shape)

# 2. Signal energy
feature_cols = [c for c in df.columns if c != "label"]
df["signal_energy"] = df[feature_cols].abs().mean(axis=1)

# 3. Thresholds
neg_energy = df[df["label"] == "NEGATIVE"]["signal_energy"]
low_thr = neg_energy.quantile(0.33)
mid_thr = neg_energy.quantile(0.66)
print(f"Thresholds: low={low_thr:.4f}, mid={mid_thr:.4f}")

# 4. Map to 5 states
def map_state(row):
    label, energy = row["label"], row["signal_energy"]
    if label == "POSITIVE": return "Focus"
    elif label == "NEUTRAL": return "Relaxation"
    elif label == "NEGATIVE":
        if energy >= mid_thr: return "Stress"
        elif energy >= low_thr: return "Fatigue"
        else: return "Drowsiness"

df["CognitiveState"] = df.apply(map_state, axis=1)
print("\nClass distribution:\n", df["CognitiveState"].value_counts())

# 5. Encode
le = LabelEncoder()
df["CognitiveState"] = le.fit_transform(df["CognitiveState"])
print("Classes:", le.classes_)

# 6. Prepare features
X = df.drop(["label", "CognitiveState", "signal_energy"], axis=1)
y = df["CognitiveState"]
X = X.replace("-", np.nan).apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())
print(f"✅ Features cleaned. NaN remaining: {X.isna().sum().sum()}")

# 7. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 9. Train
model = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# 10. Evaluate
y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.