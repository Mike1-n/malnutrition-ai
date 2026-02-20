import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def append_data(num_new_children=200, visits_per_child=5, data_file='malnutrition_data.csv'):
    """Generates and appends synthetic data for new children."""
    
    np.random.seed(42)  # For reproducibility (though appending might need a different seed or state if run multiple times)
    # Actually, to avoid identical data if run multiple times, we might want to NOT seed, or seed with something variable.
    # But for now, let's stick to a seed for consistent testing, or maybe just remove it if we want variety on multiple runs.
    # Let's remove the fixed seed for the append script to ensure variety if run multiple times.
    # np.random.seed(42) 

    if os.path.exists(data_file):
        try:
            existing_df = pd.read_csv(data_file)
            if not existing_df.empty:
                max_id = existing_df['child_id'].max()
                print(f"Found existing data. Max child_id: {max_id}")
            else:
                max_id = 0
                print("Data file exists but is empty. Starting from child_id 1.")
        except Exception as e:
            print(f"Error reading {data_file}: {e}")
            return
    else:
        max_id = 0
        print(f"{data_file} not found. Creating new file.")

    new_data = []
    
    start_date = datetime(2023, 1, 1) # Or maybe use today's date? Let's stick to the existing pattern or maybe recent dates?
    # The existing data seems to be from 2023. Let's continue with that or maybe shift to 2024?
    # Let's stick to 2023 to be consistent with the *existing* data in the file we saw earlier (2023 dates).
    
    # We want 200 *new* children.
    start_id = max_id + 1
    end_id = start_id + num_new_children

    print(f"Generating data for children IDs {start_id} to {end_id - 1}...")

    for child_id in range(start_id, end_id):
        # Baseline for child
        base_weight = np.random.uniform(5, 12)
        base_height = np.random.uniform(60, 80)
        # Gender
        gender = np.random.choice(['Male', 'Female'])
        
        # Static Socio/Medical Features (per child)
        birth_weight = round(np.random.uniform(2.5, 4.5), 2)
        income_level = np.random.choice(['low', 'middle', 'high'], p=[0.4, 0.4, 0.2])
        education_level = np.random.choice(['none', 'primary', 'secondary', 'tertiary'], p=[0.1, 0.3, 0.4, 0.2])
        water_access = np.random.choice(['yes', 'no'], p=[0.7, 0.3])
        sanitation_access = np.random.choice(['yes', 'no'], p=[0.6, 0.4])
        
        hiv_exposure = np.random.choice(
            ['hiv_unexposed', 'hiv_exposed_unaffected', 'hiv_infected', 'unknown'], 
            p=[0.85, 0.1, 0.03, 0.02]
        )
        chronic_illness = np.random.choice(['yes', 'no'], p=[0.1, 0.9])
        congenital_disease = np.random.choice(['yes', 'no'], p=[0.05, 0.95])
        recurrent_diarrhea = np.random.choice(['yes', 'no'], p=[0.2, 0.8])
        breastfeeding_6m = np.random.choice(['yes', 'no'], p=[0.6, 0.4])
        
        # Immunization Status
        immunization_status = np.random.choice(['age_appropriate', 'fully_immunized', 'partially_immunized', 'zero_dose'], 
                                             p=[0.5, 0.3, 0.15, 0.05])

        current_date = start_date
        
        for visit in range(visits_per_child):
            # Simulate growth/change over time
            weight = base_weight + (visit * growth_rate_w) + np.random.normal(0, 0.2)
            height = base_height + (visit * growth_rate_h) + np.random.normal(0, 0.5)
            # muac = base_muac + np.random.normal(0, 0.1) # MUAC removed
            
            # Simulate illness event (diarrhea, fever etc.)
            illness = np.random.choice(['yes', 'no'], p=[0.2, 0.8])
            illness_count = np.random.randint(1, 4) if illness == 'yes' else 0
            
            # Feeding features
            feeding_diversity = np.random.randint(1, 8)
            meal_freq = np.random.randint(1, 7)
            
            # Food types string
            food_options = ["carbs", "proteins", "vitamins", "fats", "dairy", "fruits", "vegetables"]
            selected_foods = np.random.choice(food_options, size=min(feeding_diversity, len(food_options)), replace=False)
            food_types_str = ",".join(selected_foods)
            
            # If ill, maybe lose weight
            if illness == 'yes':
                weight -= np.random.uniform(0.2, 0.5)

            # Determine malnutrition status (target)
            # Synthetic Z-score approximation:
            hm = height / 100.0
            bmi = weight / (hm ** 2)
            z_score = (bmi - 16) / 1.0 # Rough proxy
            
            is_malnourished = 0
            if z_score < -2:
                is_malnourished = 1
            
            # Add noise to target based on risk factors
            risk_score = 0
            if income_level == 'low': risk_score += 1
            if water_access == 'no': risk_score += 1
            if chronic_illness == 'yes': risk_score += 1
            if breastfeeding_6m == 'no': risk_score += 1
            
            # Immunization Risk
            if immunization_status == 'zero_dose': risk_score += 2
            elif immunization_status == 'partially_immunized': risk_score += 1
            
            # HIV Risk
            if hiv_exposure == 'hiv_infected': risk_score += 2
            elif hiv_exposure == 'hiv_exposed_unaffected': risk_score += 1
            
            if risk_score > 2 and np.random.random() < 0.2:
                is_malnourished = 1 # Higher risk if poor socio/health
            
            # Add noise to target
            if np.random.random() < 0.05:
                is_malnourished = 1 - is_malnourished

            new_data.append({
                'child_id': child_id,
                'gender': gender,
                'birth_weight': birth_weight,
                'visit_date': current_date.strftime('%Y-%m-%d'),
                'weight': round(weight, 2),
                'height': round(height, 2),
                'WHZ': round(z_score, 2),
                'illness': illness,
                'illness_count_last_month': illness_count,
                'recurrent_diarrhea': recurrent_diarrhea,
                'chronic_illness': chronic_illness,
                'congenital_disease': congenital_disease,
                'hiv_exposure': hiv_exposure,
                'immunization_status': immunization_status,
                'exclusive_breastfeeding_6m': breastfeeding_6m,
                'feeding_diversity_score': feeding_diversity,
                'meal_frequency_per_day': meal_freq,
                'food_types': food_types_str,
                'household_income_level': income_level,
                'parent_education_level': education_level,
                'access_to_clean_water': water_access,
                'sanitation_access': sanitation_access,
                'malnutrition': is_malnourished
            })
            
            # Next visit ~1 month later
            current_date += timedelta(days=30)

    if new_data:
        new_df = pd.DataFrame(new_data)
        
        # Append to file
        # If file didn't exist, write with header. If it did, append without header.
        mode = 'a' if os.path.exists(data_file) else 'w'
        header = not os.path.exists(data_file)
        
        new_df.to_csv(data_file, mode=mode, header=header, index=False)
        print(f"Successfully appended {len(new_df)} records for {num_new_children} children to {data_file}.")
    else:
        print("No data generated.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Append synthetic malnutrition data.')
    parser.add_argument('--num_children', type=int, default=200, help='Number of new children to generate')
    parser.add_argument('--visits', type=int, default=5, help='Number of visits per child')
    
    args = parser.parse_args()
    
    append_data(num_new_children=args.num_children, visits_per_child=args.visits)
