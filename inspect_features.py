import joblib
try:
    cols = joblib.load("osis_features.pkl")
    print("Feature Columns:", cols)
except Exception as e:
    print("Error:", e)
