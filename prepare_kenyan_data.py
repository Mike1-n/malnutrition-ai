"""
prepare_kenyan_data.py
----------------------
Generates a clinically-consistent dataset of 5,000 child records
based on the official WHO Growth Standards (LMS curves) in who_standards.csv.
The target labels and Z-scores are mathematically aligned with growth indicators.
"""
import pandas as pd
import numpy as np
import os
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Load WHO growth standards
standards_path = "who_standards.csv"
if not os.path.exists(standards_path):
    raise FileNotFoundError(f"WHO standards file not found: {standards_path}")
who_standards = pd.read_csv(standards_path)

print(f"Loaded WHO standards from {standards_path}")

num_records = 6000
data = []

# Illnesses options
illnesses = ["Fever", "Malaria", "Pneumonia", "Gastroenteritis"]

for i in range(1, num_records + 1):
    subject_id = f"K{i:04d}"
    gender = random.choice(["Male", "Female"])
    age_months = random.randint(0, 59)
    
    # 1. Height-for-age trajectory (starting ~50 cm up to ~110 cm, with biological variance)
    # Mean height follows a power curve based on age
    mean_height = 50.0 + 7.8 * (age_months ** 0.55)
    height = round(mean_height + np.random.normal(0, 2.2), 1)
    
    # Clamp height to WHO standard range (45cm to 120cm)
    height = max(45.0, min(120.0, height))
    
    # 2. Assign socio-economic and clinical factors
    income_level = str(np.random.choice(["low", "middle", "high"], p=[0.4, 0.4, 0.2]))
    
    if income_level == "low":
        parent_edu = str(np.random.choice(["none", "primary", "secondary"], p=[0.25, 0.55, 0.20]))
        clean_water = str(np.random.choice(["yes", "no"], p=[0.35, 0.65]))
        sanitation = str(np.random.choice(["yes", "no"], p=[0.30, 0.70]))
    elif income_level == "middle":
        parent_edu = str(np.random.choice(["primary", "secondary", "tertiary"], p=[0.20, 0.60, 0.20]))
        clean_water = str(np.random.choice(["yes", "no"], p=[0.80, 0.20]))
        sanitation = str(np.random.choice(["yes", "no"], p=[0.75, 0.25]))
    else: # high income
        parent_edu = str(np.random.choice(["secondary", "tertiary"], p=[0.30, 0.70]))
        clean_water = str(np.random.choice(["yes", "no"], p=[0.95, 0.05]))
        sanitation = str(np.random.choice(["yes", "no"], p=[0.90, 0.10]))
        
    hiv_exposure = str(np.random.choice(
        ["hiv_unexposed", "hiv_exposed_unaffected", "hiv_infected", "unknown"],
        p=[0.88, 0.08, 0.03, 0.01]
    ))
    
    chronic_illness = str(np.random.choice(["yes", "no"], p=[0.05, 0.95]))
    
    # Recurrent diarrhea is highly dependent on clean water and sanitation
    if clean_water == "no" and sanitation == "no":
        recurrent_diarrhea = str(np.random.choice(["yes", "no"], p=[0.45, 0.55]))
    elif clean_water == "no" or sanitation == "no":
        recurrent_diarrhea = str(np.random.choice(["yes", "no"], p=[0.25, 0.75]))
    else:
        recurrent_diarrhea = str(np.random.choice(["yes", "no"], p=[0.07, 0.93]))
        
    # Recent illness probability
    if recurrent_diarrhea == "yes" or chronic_illness == "yes":
        recent_illness_prob = 0.40
    else:
        recent_illness_prob = 0.15
        
    if random.random() < recent_illness_prob:
        recent_illness = random.choice(illnesses)
    else:
        recent_illness = "no"
        
    # Birth Weight: Shifted down by low SES or maternal HIV infection
    if income_level == "low" or hiv_exposure == "hiv_infected":
        mean_bw = 2.7
        sd_bw = 0.5
    else:
        mean_bw = 3.25
        sd_bw = 0.4
        
    birth_weight = round(max(1.0, min(5.5, np.random.normal(mean_bw, sd_bw))), 2)
    
    # Exclusive Breastfeeding: dependent on maternal education
    if parent_edu in ["secondary", "tertiary"]:
        ebf_6m = str(np.random.choice(["yes", "no"], p=[0.75, 0.25]))
    else:
        ebf_6m = str(np.random.choice(["yes", "no"], p=[0.55, 0.45]))
        
    # Immunization status (age-dependent)
    if age_months >= 12:
        immunization_status = str(np.random.choice(
            ["fully_immunized", "partially_immunized", "zero_dose"],
            p=[0.82, 0.14, 0.04]
        ))
    else:
        immunization_status = str(np.random.choice(
            ["age_appropriate", "partially_immunized", "zero_dose"],
            p=[0.85, 0.11, 0.04]
        ))
        
    # Feeding practice mapping
    # Simplified diversity score for model category proxy
    diversity = random.randint(1, 8) if age_months >= 6 else 0
    if age_months < 6:
        feeding_practice = "Exclusive Breastfeeding" if ebf_6m == "yes" else "Mixed Feeding"
    else:
        feeding_practice = "Mixed Feeding" if diversity > 3 else "Complementary Feeding"

    # 3. Model Z-score using clinical correlations
    # Base healthy child has a mean Z-score of +0.5 (representing standard low-risk growth)
    z_mean = 0.4
    
    # Risk deductions
    if birth_weight < 2.5:
        z_mean -= 0.5
    if income_level == "low":
        z_mean -= 0.6
    elif income_level == "middle":
        z_mean -= 0.2
    if clean_water == "no":
        z_mean -= 0.3
    if sanitation == "no":
        z_mean -= 0.3
    if hiv_exposure == "hiv_infected":
        z_mean -= 1.6
    elif hiv_exposure == "hiv_exposed_unaffected":
        z_mean -= 0.4
    if chronic_illness == "yes":
        z_mean -= 0.5
    if recurrent_diarrhea == "yes":
        z_mean -= 0.8
    if recent_illness != "no":
        z_mean -= 0.4
    if ebf_6m == "no" and age_months >= 6:
        z_mean -= 0.4
    if immunization_status == "zero_dose":
        z_mean -= 0.5
    elif immunization_status == "partially_immunized":
        z_mean -= 0.2
        
    # Draw child's Z-score from normal distribution around z_mean
    z_score = np.random.normal(z_mean, 0.55)
    
    # Clamp Z-score between -4.5 and +3.5 to match WHO reference scale
    z_score = max(-4.5, min(3.5, z_score))
    
    # 4. Mathematically back-calculate weight using WHO growth standards for this height and gender
    df_g = who_standards[who_standards['gender'] == gender]
    nearest_row = df_g.iloc[(df_g['height'] - height).abs().argsort()[:1]]
    L, M, S = nearest_row['L'].values[0], nearest_row['M'].values[0], nearest_row['S'].values[0]
    
    # W = M * (1 + L * S * Z)^(1/L)
    weight = M * ((1.0 + L * S * z_score) ** (1.0 / L))
    weight = round(max(2.0, min(30.0, weight)), 1)
    
    # Recalculate z_score from rounded weight/height to ensure 100% data consistency
    z_score = ((weight / M)**L - 1) / (L * S)
    z_score = round(max(-4.5, min(3.5, z_score)), 2)
    
    # 5. Calculate vulnerability points for clinical and environmental risk factors
    vuln_points = 0
    if birth_weight < 2.5:
        vuln_points += 1
    if income_level == "low":
        vuln_points += 2
    elif income_level == "middle":
        vuln_points += 1
    if clean_water == "no" or sanitation == "no":
        vuln_points += 1
    if hiv_exposure == "hiv_infected":
        vuln_points += 3
    elif hiv_exposure == "hiv_exposed_unaffected":
        vuln_points += 1
    if chronic_illness == "yes":
        vuln_points += 1
    if recurrent_diarrhea == "yes":
        vuln_points += 1
    if recent_illness != "no":
        vuln_points += 1
    if ebf_6m == "no" and age_months >= 6:
        vuln_points += 1
    if immunization_status == "zero_dose":
        vuln_points += 2
    elif immunization_status == "partially_immunized":
        vuln_points += 1
        
    # Multi-parametric target classification rules:
    # A child is physically malnourished (Moderate/Severe) ONLY if their Z-score is <= -2.0 SD.
    # Otherwise, if their Z-score is > -2.0 SD but they have high clinical/environmental risks, 
    # they are classified as having a high risk of getting malnutrition (High Risk).
    if z_score <= -3.0:
        malnutrition_category = "Severe Malnutrition"
    elif z_score <= -2.0:
        if vuln_points >= 6 or hiv_exposure == "hiv_infected":
            malnutrition_category = "Severe Malnutrition"
        else:
            malnutrition_category = "Moderate Malnutrition"
    else:
        if vuln_points >= 3 or z_score <= -1.0:
            malnutrition_category = "High Risk"
        elif vuln_points >= 2:
            malnutrition_category = "Moderate Risk"
        else:
            malnutrition_category = "Low Risk"
        
    data.append({
        "subject_id": subject_id,
        "age_months": age_months,
        "gender": gender,
        "birth_weight": birth_weight,
        "weight": weight,
        "height": height,
        "recent_illness": recent_illness,
        "chronic_illness": chronic_illness,
        "immunization_status": immunization_status,
        "feeding_practice": feeding_practice,
        "household_income_level": income_level,
        "parent_education_level": parent_edu,
        "access_to_clean_water": clean_water,
        "sanitation_access": sanitation,
        "hiv_exposure": hiv_exposure,
        "recurrent_diarrhea": recurrent_diarrhea,
        "exclusive_breastfeeding_6m": ebf_6m,
        "z_score": z_score,
        "malnutrition_category": malnutrition_category,
    })

# Convert to DataFrame
df_out = pd.DataFrame(data)

# Save to CSV
out_path = "kenyan_malnutrition_data.csv"
df_out.to_csv(out_path, index=False)

print(f"\nSaved {len(df_out)} rows to {out_path}")
print("\nCategory distribution:")
print(df_out["malnutrition_category"].value_counts())
print("\nSample rows:")
print(df_out.head(3).to_string())
