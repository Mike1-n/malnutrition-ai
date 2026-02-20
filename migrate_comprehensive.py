import pandas as pd
import numpy as np
import os

def migrate_comprehensive():
    print("Loading data...")
    if not os.path.exists('malnutrition_data.csv'):
        print("malnutrition_data.csv not found.")
        return

    df = pd.read_csv('malnutrition_data.csv')
    
    # Check if we already have some of these columns to avoid overwriting if run multiple times?
    # For now, let's assume we want to ensure they exist and are populated.
    
    child_ids = df['child_id'].unique()
    
    print(f"Backfilling comprehensive data for {len(child_ids)} children...")
    
    # We need to assign static attributes per CHILD, not per visit.
    
    # 1. Birth Weight (requested earlier)
    # Range: 2.5 to 4.5 kg typically
    birth_weights = {cid: round(np.random.uniform(2.5, 4.5), 2) for cid in child_ids}
    
    # 2. Socio-Economic
    # household_income_level: ["low", "middle", "high"]
    incomes = {cid: np.random.choice(['low', 'middle', 'high'], p=[0.4, 0.4, 0.2]) for cid in child_ids}
    
    # parent_education_level: ["none", "primary", "secondary", "tertiary"]
    educations = {cid: np.random.choice(['none', 'primary', 'secondary', 'tertiary'], p=[0.1, 0.3, 0.4, 0.2]) for cid in child_ids}
    
    # access_to_clean_water: ["yes", "no"]
    water_access = {cid: np.random.choice(['yes', 'no'], p=[0.7, 0.3]) for cid in child_ids}
    
    # sanitation_access: ["yes", "no"]
    sanitation = {cid: np.random.choice(['yes', 'no'], p=[0.6, 0.4]) for cid in child_ids}
    
    # 3. Medical / Biological (Static/Long-term)
    # hiv_exposure: ["hiv_unexposed", "hiv_exposed_unaffected", "hiv_infected", "unknown"]
    hiv_status = {cid: np.random.choice(['hiv_unexposed', 'hiv_exposed_unaffected', 'hiv_infected', 'unknown'], 
                               p=[0.85, 0.1, 0.03, 0.02]) for cid in child_ids}
    
    # chronic_illness: ["yes", "no"]
    chronic = {cid: np.random.choice(['yes', 'no'], p=[0.1, 0.9]) for cid in child_ids}
    
    # congenital_disease: ["yes", "no"]
    congenital = {cid: np.random.choice(['yes', 'no'], p=[0.05, 0.95]) for cid in child_ids}
    
    # recurrent_diarrhea: ["yes", "no"]
    diarrhea_hist = {cid: np.random.choice(['yes', 'no'], p=[0.2, 0.8]) for cid in child_ids}
    
    # exclusive_breastfeeding_6m: ["yes", "no"]
    breastfeeding = {cid: np.random.choice(['yes', 'no'], p=[0.6, 0.4]) for cid in child_ids}
    
    # immunization_status: ["age_appropriate", "fully_immunized", "partially_immunized", "zero_dose"]
    # For backfilling, we use 'age_appropriate' as the safe "up-to-date" status for all ages.
    # We avoid 'fully_immunized' blindly to prevent assigning it to infants < 12m.
    immunization = {cid: np.random.choice(['age_appropriate', 'partially_immunized', 'zero_dose'], 
                                        p=[0.8, 0.15, 0.05]) for cid in child_ids}
    
    # Apply Static Features
    df['birth_weight'] = df['child_id'].map(birth_weights)
    df['household_income_level'] = df['child_id'].map(incomes)
    df['parent_education_level'] = df['child_id'].map(educations)
    df['access_to_clean_water'] = df['child_id'].map(water_access)
    df['sanitation_access'] = df['child_id'].map(sanitation)
    df['hiv_exposure'] = df['child_id'].map(hiv_status)
    df['chronic_illness'] = df['child_id'].map(chronic)
    df['congenital_disease'] = df['child_id'].map(congenital)
    df['recurrent_diarrhea'] = df['child_id'].map(diarrhea_hist)
    df['exclusive_breastfeeding_6m'] = df['child_id'].map(breastfeeding)
    df['immunization_status'] = df['child_id'].map(immunization)
    
    # 4. Dynamic/Visit-based Features
    # These might change per visit, but we can also generate them randomly per row for simplicity in synthetic data,
    # or keep them somewhat consistent per child. Let's make them per-row (visit) to show variability.
    
    # feeding_diversity_score: 1-7 (integer)
    df['feeding_diversity_score'] = np.random.randint(1, 8, size=len(df))
    
    # meal_frequency_per_day: integer (1-6 usually)
    df['meal_frequency_per_day'] = np.random.randint(1, 7, size=len(df))
    
    # illness_count_last_month: integer (0-5)
    # We already have 'illness' (yes/no). Let's correlate them?
    # If illness=yes, count is likely >= 1. If no, likely 0.
    df['illness_count_last_month'] = df.apply(
        lambda row: np.random.randint(1, 4) if row['illness'] == 'yes' else 0, 
        axis=1
    )
    
    # food_types: ["carbs", "proteins", "vitamins", "fats"]
    # This is a multi-select. In CSV, maybe comma-separated string? Or just ignored for ML if we use diversity score.
    # The user request lists it, but for ML, 'feeding_diversity_score' is usually the aggregate metric derived from this.
    # Let's add it as a string representation for display purposes.
    def gen_food_types(diversity):
        types = ["carbs", "proteins", "vitamins", "fats", "dairy", "fruits", "vegetables"]
        # Select 'diversity' number of items
        selected = np.random.choice(types, size=diversity, replace=False) if diversity <= len(types) else types
        return ",".join(selected)

    df['food_types'] = df['feeding_diversity_score'].apply(gen_food_types)

    print("Saving enriched dataset...")
    df.to_csv('malnutrition_data.csv', index=False)
    print("Migration complete. Dataset enriched with comprehensive WHO/clinical indicators.")

if __name__ == "__main__":
    migrate_comprehensive()
