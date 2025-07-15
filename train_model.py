import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# ----------- very small sample dataset -----------
data = {
    "sleep":       [8, 5, 6, 7, 4, 3, 7, 6, 9, 2],
    "stress":      [2, 7, 6, 4, 8, 9, 3, 6, 1, 10],
    "depression":  [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    "appetite":    [0, 1, 1, 0, 1, 1, 0, 0, 0, 1],
    "focus":       [0, 1, 1, 0, 1, 1, 0, 0, 0, 1],
    "status":      ["Healthy ", "Needs Attention ", "Mild Stress ", "Healthy ",
                    "Needs Attention ", "Needs Attention ", "Healthy ", "Mild Stress ",
                    "Healthy ", "Needs Attention "]
}

df = pd.DataFrame(data)

X = df.drop("status", axis=1)
y = df["status"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/mental_health_model.pkl")

print("✅  Model saved →  model/mental_health_model.pkl")
