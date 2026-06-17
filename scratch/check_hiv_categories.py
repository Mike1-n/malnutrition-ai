import joblib
import pandas as pd
import os

print("--- Checking malnutrition_data.csv ---")
if os.path.exists("malnutrition_data.csv"):
    df = pd.read_csv("malnutrition_data.csv", nrows=10)
    if "hiv_exposure" in df.columns:
        print("Unique values in data:", pd.read_csv("malnutrition_data.csv")["hiv_exposure"].dropna().unique())

print("--- Checking malnutrition_model.pkl ---")
if os.path.exists("malnutrition_model.pkl"):
    try:
        model = joblib.load("malnutrition_model.pkl")
        # Check feature names and their categories if it's a pipeline or encoder
        print("Model object:", type(model))
        # If the model has encoder steps
        if hasattr(model, 'feature_names_in_'):
            print("Feature names:", model.feature_names_in_)
    except Exception as e:
        print("Error reading model:", e)
