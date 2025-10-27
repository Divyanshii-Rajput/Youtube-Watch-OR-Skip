import joblib

data = joblib.load("backend/model.joblib")

# Extract the actual classifier
model = data.get("clf", None)
if model is None:
    raise ValueError("No classifier found under 'clf' key!")

# Save it cleanly
joblib.dump(model, "backend/model_clean.joblib")

print("✅ Clean RandomForest model saved at backend/model_clean.joblib")
