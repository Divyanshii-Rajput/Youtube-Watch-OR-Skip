# backend/train_model.py
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

os.makedirs("data", exist_ok=True)

# Load dataset
df = pd.read_csv("data/youtube_dataset.csv")
print("📊 Loaded dataset:", df.shape)

X = df[["log_views", "likes", "comment_count", "like_ratio", "sentiment"]]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(random_state=42))
])

param_grid = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [6, 10, None],
    "rf__min_samples_leaf": [2, 4],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring="accuracy", verbose=1)
grid.fit(X_train, y_train)

print("✅ Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Final Test Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(best_model, "backend/model_pipeline.joblib")
print("✅ Model saved successfully!")

# Feature Importance
rf = best_model.named_steps['rf']
importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(7, 4))
sns.barplot(x=features, y=importances, color="#ffb6c1")
plt.title("Feature Importance in Watch-or-Skip Prediction")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("data/feature_importance.png")
plt.close()


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Skip", "Watch"], yticklabels=["Skip", "Watch"])
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("data/confusion_matrix.png")
plt.close()

print("📈 Charts saved to data/ folder.")
