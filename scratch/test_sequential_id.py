import re
import requests
import os
import toml
import pandas as pd

# Load credentials
secrets_path = os.path.join(".streamlit", "secrets.toml")
if not os.path.exists(secrets_path):
    print("secrets.toml not found")
    exit(1)

secrets = toml.load(secrets_path)
sb_url = secrets.get("SUPABASE_URL", "").strip('"')
sb_key = secrets.get("SUPABASE_KEY", "").strip('"')

def get_unique_subject_id_test():
    max_id_num = 0
    pattern = re.compile(r'^M(\d+)$', re.IGNORECASE)
    
    # 1. Try to find the max ID from Supabase if connected
    if sb_url and sb_key:
        headers = {
            "apikey": sb_key,
            "Authorization": f"Bearer {sb_key}",
            "Content-Type": "application/json"
        }
        # Order by subject_id descending and limit to 1 to quickly find the latest record
        endpoint = f"{sb_url.rstrip('/')}/rest/v1/assessments?select=subject_id&order=subject_id.desc&limit=1"
        try:
            response = requests.get(endpoint, headers=headers, timeout=5)
            if response.status_code == 200:
                records = response.json()
                if records:
                    sid = records[0].get("subject_id")
                    if sid:
                        match = pattern.match(str(sid).strip())
                        if match:
                            max_id_num = int(match.group(1))
                            print("Supabase max subject ID found:", sid)
        except Exception as e:
            print("Supabase query failed:", e)

    # 2. Check local CSV files to see if there are higher IDs (e.g., up to M001)
    for csv_file in ["kenyan_malnutrition_data.csv", "malnutrition_data.csv"]:
        try:
            if os.path.exists(csv_file):
                header = pd.read_csv(csv_file, nrows=0).columns.tolist()
                if "subject_id" in header:
                    df_temp = pd.read_csv(csv_file, usecols=["subject_id"])
                    for sid in df_temp["subject_id"].dropna().unique():
                        match = pattern.match(str(sid).strip())
                        if match:
                            num = int(match.group(1))
                            if num > max_id_num:
                                max_id_num = num
        except Exception as e:
            print(f"CSV {csv_file} check failed:", e)

    next_num = max_id_num + 1
    return f"M{next_num:03d}"

print("Generated Subject ID:", get_unique_subject_id_test())
