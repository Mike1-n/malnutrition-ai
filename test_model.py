import pandas as pd
import joblib

model = joblib.load('malnutrition_model.pkl')

input_vector = pd.DataFrame([{
    'age_months': 24,
    'weight': 9.5,
    'height': 75.0,
    'muac_mm': 135.0,
    'gender': 'Male',
    'birth_weight': 3.0,
    'household_income_level': 'middle',
    'recent_illness': 'no',
    'chronic_illness': 'no',
    'immunization_status': 'fully_immunized',
    'feeding_practice': 'Mixed Feeding',
}])

prob = model.predict_proba(input_vector)
print("Prediction probability:", prob)
