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

# Test ordering by subject_id desc, limit 1
endpoint_subject_id = f"{url.rstrip('/')}/rest/v1/assessments?select=subject_id&order=subject_id.desc&limit=1"
print(f"Testing endpoint: {endpoint_subject_id}")
try:
    response = requests.get(endpoint_subject_id, headers=headers, timeout=5)
    print("Status code:", response.status_code)
    print("Response body:", response.json())
except Exception as e:
    print("Query failed:", e)

# Test ordering by id desc, limit 1
endpoint_id = f"{url.rstrip('/')}/rest/v1/assessments?select=subject_id&order=id.desc&limit=1"
print(f"Testing endpoint: {endpoint_id}")
try:
    response = requests.get(endpoint_id, headers=headers, timeout=5)
    print("Status code:", response.status_code)
    print("Response body:", response.json())
except Exception as e:
    print("Query failed:", e)
