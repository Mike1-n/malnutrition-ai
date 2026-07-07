import time
import requests
import os
import toml

secrets_path = os.path.join(".streamlit", "secrets.toml")
if not os.path.exists(secrets_path):
    print("secrets.toml not found")
    exit(1)

secrets = toml.load(secrets_path)
url = secrets.get("SUPABASE_URL", "").strip('"')
key = secrets.get("SUPABASE_KEY", "").strip('"')

headers = {
    "apikey": key,
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}

# 1. Test count of assessments
count_endpoint = f"{url.rstrip('/')}/rest/v1/assessments?select=id"
print(f"Testing assessments count from: {count_endpoint}")
start = time.time()
try:
    response = requests.get(count_endpoint, headers=headers, timeout=30)
    elapsed = time.time() - start
    if response.status_code == 200:
        records = response.json()
        print(f"Success! Fetched {len(records)} records in {elapsed:.4f}s")
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print("Failed count query:", e)

# 2. Test full fetch of assessments (what get_all_assessments does)
full_endpoint = f"{url.rstrip('/')}/rest/v1/assessments?select=*&order=id.desc"
print(f"Testing full assessments query from: {full_endpoint}")
start = time.time()
try:
    response = requests.get(full_endpoint, headers=headers, timeout=30)
    elapsed = time.time() - start
    if response.status_code == 200:
        records = response.json()
        print(f"Success! Fetched {len(records)} full records in {elapsed:.4f}s")
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print("Failed full query:", e)
