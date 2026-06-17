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

# Let's try to query first to see if there is any pending profile
query_endpoint = f"{url.rstrip('/')}/rest/v1/user_profiles?is_approved=eq.false&select=*"
headers = {
    "apikey": key,
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

response = requests.get(query_endpoint, headers=headers)
print("Pending profiles response status:", response.status_code)
pending = response.json()
print("Pending profiles:", pending)

if pending:
    user_id = pending[0]['id']
    email = pending[0]['email']
    print(f"Attempting to approve {email} (ID: {user_id})")
    
    # Try updating
    update_endpoint = f"{url.rstrip('/')}/rest/v1/user_profiles?id=eq.{user_id}"
    payload = {"is_approved": True}
    
    # Send request with return=representation so we see if it was modified
    headers["Prefer"] = "return=representation"
    update_response = requests.patch(update_endpoint, headers=headers, json=payload)
    print("PATCH response status:", update_response.status_code)
    try:
        print("PATCH response body:", update_response.json())
    except Exception:
        print("PATCH response text:", update_response.text)
else:
    print("No pending profiles found to test with.")
