"""
prepare_kenyan_data.py
----------------------
Reads the simulated Kenyan malnutrition dataset, maps columns to the model
schema, simulates realistic MUAC values, and saves kenyan_malnutrition_data.csv.
"""
import pandas as pd
import numpy as np
import os
import shutil

# ── 1. Locate source file ────────────────────────────────────────────────────
src = r"C:\Users\frei\Downloads\simulated_malnutrition_dataset_kenyan_proportions_v2-1.csv"
if not os.path.exists(src):
    raise FileNotFoundError(f"Source CSV not found:\n  {src}")

df = pd.read_csv(src)
print(f"Loaded {len(df)} rows from source CSV.")
print("Columns:", df.columns.tolist())

# ── 2. Column mapping ────────────────────────────────────────────────────────
df = df.rename(columns={
    "Child_ID":              "child_id",
    "Age_months":            "age_months",
    "Sex":                   "gender",
    "Birthweight_kg":        "birth_weight",
    "Weight_kg":             "weight",
    "Height_cm":             "height",
    "Recent_Illness":        "recent_illness_raw",
    "Chronic_Illness":       "chronic_illness_raw",
    "Immunization_Status":   "immunization_status_raw",
    "Feeding_Practice":      "feeding_practice",
    "Socioeconomic_Status":  "household_income_level",
    "Malnutrition_Category": "malnutrition_category",
})

# ── 3. Clean / encode columns ─────────────────────────────────────────────────

# recent_illness: "None" → "no", anything else → illness name (kept as-is for model)
df["recent_illness"] = df["recent_illness_raw"].apply(
    lambda x: "no" if str(x).strip().lower() == "none" else str(x).strip()
)

# chronic_illness: "None" → "no", anything else → "yes"
df["chronic_illness"] = df["chronic_illness_raw"].apply(
    lambda x: "no" if str(x).strip().lower() == "none" else "yes"
)

# immunization_status: Standardise to lowercase with underscores
imm_map = {
    "Fully Immunized":    "fully_immunized",
    "Partially Immunized":"partially_immunized",
    "Zero Dose":          "zero_dose",
    "Age Appropriate":    "age_appropriate",
}
df["immunization_status"] = df["immunization_status_raw"].map(imm_map).fillna("age_appropriate")

# household_income_level: capitalise → lowercase
df["household_income_level"] = df["household_income_level"].str.lower()

# ── 4. Simulate MUAC (mm) ─────────────────────────────────────────────────────
# WHO reference ranges:
#   < 115 mm → Severe Acute Malnutrition (SAM)
#   115–125 mm → Moderate Acute Malnutrition (MAM)
#   > 125 mm → Normal
# We simulate based on the Malnutrition_Category label.

np.random.seed(42)

def simulate_muac(row):
    cat = row["malnutrition_category"]
    age = row["age_months"]

    # Base MUAC increases slightly with age (saturation-like)
    age_factor = min(age / 60.0, 1.0) * 10  # up to +10 mm for 5-yr-olds

    if cat == "Severe Malnutrition":
        base = np.random.uniform(100, 114)
    elif cat == "Moderate Malnutrition":
        base = np.random.uniform(115, 124)
    elif cat == "Not Malnourished - High Risk":
        base = np.random.uniform(125, 131)
    elif cat == "Not Malnourished - Moderate Risk":
        base = np.random.uniform(132, 139)
    else:  # Low Risk / Normal
        base = np.random.uniform(140, 155)

    muac = base + age_factor + np.random.normal(0, 2.5)
    return round(np.clip(muac, 90, 180), 1)

df["muac_mm"] = df.apply(simulate_muac, axis=1)

# ── 5. Select and order final columns ────────────────────────────────────────
keep_cols = [
    "child_id",
    "age_months",
    "gender",
    "birth_weight",
    "weight",
    "height",
    "muac_mm",
    "recent_illness",
    "chronic_illness",
    "immunization_status",
    "feeding_practice",
    "household_income_level",
    "malnutrition_category",
]
df_out = df[keep_cols].copy()

# ── 6. Save ───────────────────────────────────────────────────────────────────
out_path = "kenyan_malnutrition_data.csv"
df_out.to_csv(out_path, index=False)
print(f"\nSaved {len(df_out)} rows → {out_path}")
print("\nCategory distribution:")
print(df_out["malnutrition_category"].value_counts())
print("\nMUAC stats (mm):")
print(df_out["muac_mm"].describe().round(1))
print("\nSample rows:")
print(df_out.head(3).to_string())
