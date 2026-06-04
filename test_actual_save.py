import sys
import os
import toml

sys.path.append(os.path.abspath("."))
import database as db

# Load credentials from secrets.toml
secrets_path = os.path.join(".streamlit", "secrets.toml")
if not os.path.exists(secrets_path):
    print("secrets.toml not found")
    sys.exit(1)

secrets = toml.load(secrets_path)
url = secrets.get("SUPABASE_URL", "").strip('"')
key = secrets.get("SUPABASE_KEY", "").strip('"')

print(f"Connecting to URL: {url}")
# Use a simple dummy payload
dummy_payload = {
    "subject_id": "TEST_SUBJECT",
    "age_months": 24,
    "gender": "Male",
    "birth_weight": 3.0,
    "weight": 10.0,
    "height": 80.0,
    "recent_illness": "no",
    "chronic_illness": "no",
    "immunization_status": "fully_immunized",
    "feeding_practice": "Mixed Feeding",
    "household_income_level": "middle",
    "parent_education_level": "secondary",
    "access_to_clean_water": "yes",
    "sanitation_access": "yes",
    "hiv_exposure": "hiv_unexposed",
    "recurrent_diarrhea": "no",
    "exclusive_breastfeeding_6m": "yes",
    "feeding_diversity_score": 4,
    "ses_score": 3,
    "z_score": 0.0,
    "whz_category": "Normal",
    "ml_risk": "Low Risk",
    "ml_confidence": 0.95
}

success, msg = db.save_assessment(url, key, dummy_payload)
print(f"Success: {success}")
print(f"Message: {msg}")
