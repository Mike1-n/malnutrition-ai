import pandas as pd
import joblib

model = joblib.load('malnutrition_model.pkl')

profiles = [
    # 1. Healthy / Low Risk (Z-Score > 1.0, 0 risks)
    {
        'name': 'Low Risk Profile (Z > 1.0)',
        'age_months': 24, 'weight': 13.0, 'height': 85.0, 'gender': 'Male',
        'birth_weight': 3.5, 'household_income_level': 'high', 'parent_education_level': 'tertiary',
        'access_to_clean_water': 'yes', 'sanitation_access': 'yes', 'hiv_exposure': 'hiv_unexposed',
        'chronic_illness': 'no', 'recurrent_diarrhea': 'no', 'exclusive_breastfeeding_6m': 'yes',
        'immunization_status': 'fully_immunized', 'feeding_practice': 'Mixed Feeding', 'recent_illness': 'no',
        'z_score': 1.20
    },
    # 2. Normal / Moderate Risk (Z = -0.08, 0 risks)
    {
        'name': 'Moderate Risk Profile (Z = -0.08, 0 risks)',
        'age_months': 24, 'weight': 10.5, 'height': 85.0, 'gender': 'Male',
        'birth_weight': 3.2, 'household_income_level': 'high', 'parent_education_level': 'tertiary',
        'access_to_clean_water': 'yes', 'sanitation_access': 'yes', 'hiv_exposure': 'hiv_unexposed',
        'chronic_illness': 'no', 'recurrent_diarrhea': 'no', 'exclusive_breastfeeding_6m': 'yes',
        'immunization_status': 'fully_immunized', 'feeding_practice': 'Mixed Feeding', 'recent_illness': 'no',
        'z_score': -0.08
    },
    # 3. Vulnerable Child with Normal Z-Score (Z = -0.33, CVS = 6 risks)
    # HIV Exposed, Partially Immunized, Recent Illness (Fever), 2/8 foods, Low SES
    {
        'name': 'Vulnerable child with Normal Z-Score (Z = -0.33, CVS = 6)',
        'age_months': 24, 'weight': 10.0, 'height': 85.0, 'gender': 'Male',
        'birth_weight': 3.0, 'household_income_level': 'low', 'parent_education_level': 'primary',
        'access_to_clean_water': 'no', 'sanitation_access': 'no', 'hiv_exposure': 'hiv_exposed_unaffected',
        'chronic_illness': 'no', 'recurrent_diarrhea': 'no', 'exclusive_breastfeeding_6m': 'yes',
        'immunization_status': 'partially_immunized', 'feeding_practice': 'Complementary Feeding', 'recent_illness': 'yes',
        'z_score': -0.33
    },
    # 4. Moderate Malnutrition (-3.0 < Z <= -2.0)
    {
        'name': 'Moderate Malnutrition Profile (-3.0 < Z <= -2.0)',
        'age_months': 24, 'weight': 8.0, 'height': 85.0, 'gender': 'Male',
        'birth_weight': 2.4, 'household_income_level': 'low', 'parent_education_level': 'none',
        'access_to_clean_water': 'no', 'sanitation_access': 'no', 'hiv_exposure': 'hiv_exposed_unaffected',
        'chronic_illness': 'no', 'recurrent_diarrhea': 'yes', 'exclusive_breastfeeding_6m': 'no',
        'immunization_status': 'partially_immunized', 'feeding_practice': 'Complementary Feeding', 'recent_illness': 'yes',
        'z_score': -2.45
    },
    # 5. Severe Malnutrition (Z <= -3.0)
    {
        'name': 'Severe Malnutrition Profile (Z <= -3.0)',
        'age_months': 24, 'weight': 6.8, 'height': 85.0, 'gender': 'Male',
        'birth_weight': 2.1, 'household_income_level': 'low', 'parent_education_level': 'none',
        'access_to_clean_water': 'no', 'sanitation_access': 'no', 'hiv_exposure': 'hiv_infected',
        'chronic_illness': 'yes', 'recurrent_diarrhea': 'yes', 'exclusive_breastfeeding_6m': 'no',
        'immunization_status': 'zero_dose', 'feeding_practice': 'Complementary Feeding', 'recent_illness': 'yes',
        'z_score': -3.60
    }
]

print("=== Running Prediction Verification ===")
for p in profiles:
    name = p.pop('name')
    input_df = pd.DataFrame([p])
    
    pred_raw = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    idx_raw = classes.index(pred_raw)
    conf_raw = probs[idx_raw]
    
    # Enforce clinical constraints
    z_score = p['z_score']
    if z_score <= -2.0:
        allowed_classes = ["Moderate Malnutrition", "Severe Malnutrition"]
    else:
        allowed_classes = ["High Risk", "Moderate Risk", "Low Risk"]
        
    allowed_probs = []
    for c in allowed_classes:
        if c in classes:
            allowed_probs.append((c, probs[classes.index(c)]))
        else:
            allowed_probs.append((c, 0.0))
            
    best_class, best_prob = max(allowed_probs, key=lambda x: x[1])
    if best_prob == 0.0:
        if z_score <= -3.0:
            pred_constrained = "Severe Malnutrition"
        elif z_score <= -2.0:
            pred_constrained = "Moderate Malnutrition"
        else:
            pred_constrained = "High Risk" if z_score <= -1.0 else "Low Risk"
        conf_constrained = 1.0
    else:
        sum_probs = sum(p for c, p in allowed_probs)
        conf_constrained = best_prob / sum_probs if sum_probs > 0 else best_prob
        pred_constrained = best_class
        
    print(f"\nProfile: {name}")
    print(f"Calculated Z-Score: {z_score} SD")
    print(f"Raw Prediction    : {pred_raw} (Confidence: {conf_raw:.1%})")
    print(f"Constrained Class : {pred_constrained} (Confidence: {conf_constrained:.1%})")
    # Print other classes with non-zero probability
    for cl, pr in zip(model.classes_, probs):
        if pr > 0.01:
            print(f"  - {cl}: {pr:.1%}")
