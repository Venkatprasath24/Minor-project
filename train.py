import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("emotions.csv")
df = df.dropna(subset=["label"])

print("Dataset loaded:", df.shape)

# Features
feature_cols = [c for c in df.columns if c != "label"]

# Encode label
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

X = df[feature_cols]
y = df["label"]

# Clean data
X = X.replace("-", np.nan)
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("Model saved successfully")