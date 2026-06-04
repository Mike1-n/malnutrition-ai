import requests
import toml
import os
import re

def get_credentials():
    url = ""
    key = ""
    
    # 1. Try reading from app.py
    if os.path.exists("app.py"):
        try:
            with open("app.py", "r", encoding="utf-8") as f:
                content = f.read()
                url_match = re.search(r'SUPABASE_URL\s*=\s*["\']([^"\']+)["\']', content)
                key_match = re.search(r'SUPABASE_KEY\s*=\s*["\']([^"\']+)["\']', content)
                if url_match:
                    url = url_match.group(1)
                if key_match:
                    key = key_match.group(1)
        except Exception as e:
            print(f"Error reading app.py: {e}")
            
    # Clean up defaults
    if url == "https://your-project.supabase.co":
        url = ""
    if key == "your-anon-key":
        key = ""
        
    # 2. Try secrets.toml if not configured in app.py
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if (not url or not key) and os.path.exists(secrets_path):
        try:
            secrets = toml.load(secrets_path)
            s_url = secrets.get("SUPABASE_URL", "")
            s_key = secrets.get("SUPABASE_KEY", "")
            if s_url and s_url != "https://your-project.supabase.co":
                url = s_url
            if s_key and s_key != "your-anon-key":
                key = s_key
        except Exception as e:
            print(f"Error reading secrets.toml: {e}")
            
    return url, key

def migrate():
    url, key = get_credentials()
    if not url or not key:
        print("[ERROR] Supabase credentials not found or still set to defaults.")
        print("Please configure SUPABASE_URL and SUPABASE_KEY in app.py first.")
        return
        
    print(f"Connecting to Supabase at: {url}")
    
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    
    mappings = {
        "Not Malnourished - High Risk": "High Risk",
        "Not Malnourished - Moderate Risk": "Moderate Risk",
        "Not Malnourished - Low Risk": "Low Risk"
    }
    
    for old_val, new_val in mappings.items():
        endpoint = f"{url.rstrip('/')}/rest/v1/assessments?ml_risk=eq.{old_val.replace(' ', '%20')}"
        payload = {"ml_risk": new_val}
        
        try:
            response = requests.patch(endpoint, headers=headers, json=payload)
            if response.status_code == 200:
                updated_count = len(response.json())
                print(f"[SUCCESS] Migrated '{old_val}' -> '{new_val}' (Updated {updated_count} records)")
            elif response.status_code == 204:
                print(f"[SUCCESS] Migrated '{old_val}' -> '{new_val}' (No content returned/already up to date)")
            else:
                print(f"[FAILED] Failed to migrate '{old_val}': HTTP {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[ERROR] Error migrating '{old_val}': {e}")

if __name__ == "__main__":
    migrate()
