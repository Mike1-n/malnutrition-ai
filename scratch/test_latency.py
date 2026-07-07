import time
import requests
import socket
import urllib.parse
import os
import toml

secrets_path = os.path.join(".streamlit", "secrets.toml")
if not os.path.exists(secrets_path):
    print("secrets.toml not found")
    exit(1)

secrets = toml.load(secrets_path)
url = secrets.get("SUPABASE_URL", "").strip('"')
key = secrets.get("SUPABASE_KEY", "").strip('"')

parsed = urllib.parse.urlparse(url)
host = parsed.netloc

print(f"Testing host: {host}")

# 1. DNS Resolution
try:
    start = time.time()
    ip = socket.gethostbyname(host)
    dns_time = time.time() - start
    print(f"DNS Resolution: {dns_time:.4f}s (IP: {ip})")
except Exception as e:
    print("DNS Resolution failed:", e)

# 2. Connection Time
try:
    start = time.time()
    s = socket.create_connection((host, 443), timeout=10)
    conn_time = time.time() - start
    s.close()
    print(f"TCP Connection to port 443: {conn_time:.4f}s")
except Exception as e:
    print("TCP Connection failed:", e)

# 3. Simple GET request
query_endpoint = f"{url.rstrip('/')}/rest/v1/user_profiles?limit=1"
headers = {
    "apikey": key,
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}

print(f"Sending GET to: {query_endpoint}")
try:
    start = time.time()
    response = requests.get(query_endpoint, headers=headers, timeout=30)
    req_time = time.time() - start
    print(f"HTTP GET Status: {response.status_code}")
    print(f"HTTP GET Time: {req_time:.4f}s")
    print(f"Response: {response.text}")
except requests.exceptions.Timeout:
    print(f"HTTP GET request timed out after 30 seconds.")
except Exception as e:
    print("HTTP GET failed:", e)
