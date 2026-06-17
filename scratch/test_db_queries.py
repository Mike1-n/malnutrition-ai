import sys
import os
import toml
import requests

secrets_path = os.path.join(".streamlit", "secrets.toml")
if not os.path.exists(secrets_path):
    print("secrets.toml not found")
    sys.exit(1)

secrets = toml.load(secrets_path)
url = secrets.get("SUPABASE_URL", "").strip('"')
key = secrets.get("SUPABASE_KEY", "").strip('"')

# Query user profiles
endpoint = f"{url.rstrip('/')}/rest/v1/user_profiles?email=eq.admin@gmail.com&select=*"
headers = {
    "apikey": key,
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}

try:
    response = requests.get(endpoint, headers=headers)
    print("User Profiles response:", response.status_code)
    print(response.json())
except Exception as e:
    print("Error querying user profiles:", e)

# Test signin
signin_endpoint = f"{url.rstrip('/')}/auth/v1/token?grant_type=password"
payload = {
    "email": "admin@gmail.com",
    "password": "123456"
}
try:
    response = requests.post(signin_endpoint, headers={"apikey": key, "Content-Type": "application/json"}, json=payload)
    print("Signin response status:", response.status_code)
    print(response.json())
except Exception as e:
    print("Error signing in:", e)
