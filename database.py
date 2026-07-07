import requests

def get_headers(key):
    """Returns headers required for Supabase authentication and REST API calls."""
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

def save_assessment(url, key, data):
    """
    Saves or updates an assessment record in the Supabase database.
    
    url: The Supabase project URL.
    key: The Supabase anon/service_role key.
    data: dict containing the assessment fields.
    
    Returns (success: bool, message: str)
    """
    endpoint = f"{url.rstrip('/')}/rest/v1/assessments?on_conflict=subject_id,age_months"
    headers = get_headers(key)
    headers["Prefer"] = "resolution=merge-duplicates"
    try:
        response = requests.post(endpoint, headers=headers, json=data, timeout=7)
        response.raise_for_status()
        return True, "Assessment saved successfully to Supabase!"
    except requests.exceptions.HTTPError as e:
        try:
            err_msg = response.json().get('message', str(e))
        except:
            err_msg = response.text or str(e)
        return False, f"Supabase Error: {err_msg}"
    except Exception as e:
        return False, f"Error connecting to database: {e}"

def get_all_assessments(url, key):
    """
    Fetches all assessments from the Supabase database.
    """
    # Order by ID or created_at descending (latest first)
    endpoint = f"{url.rstrip('/')}/rest/v1/assessments?select=*&order=id.desc"
    try:
        response = requests.get(endpoint, headers=get_headers(key), timeout=7)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching assessments: {e}")
        return []

def get_assessments_by_subject(url, key, subject_id):
    """
    Fetches historical assessments for a specific subject from Supabase.
    """
    # Order by ID or created_at ascending (chronological order for trends)
    endpoint = f"{url.rstrip('/')}/rest/v1/assessments?subject_id=eq.{subject_id}&order=id.asc"
    try:
        response = requests.get(endpoint, headers=get_headers(key), timeout=7)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching assessments for subject {subject_id}: {e}")
        return []

def signup_user(url, key, email, password):
    """
    Creates a new user account in Supabase Auth.
    """
    endpoint = f"{url.rstrip('/')}/auth/v1/signup"
    headers = {
        "apikey": key,
        "Content-Type": "application/json"
    }
    payload = {
        "email": email,
        "password": password
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=7)
        res_data = response.json()
        if response.status_code in (200, 201):
            return True, res_data
        else:
            err_msg = res_data.get('error_description') or res_data.get('msg') or res_data.get('error', {}).get('message') or response.text
            return False, err_msg
    except Exception as e:
        return False, str(e)

def signin_user(url, key, email, password):
    """
    Logs in an existing user and returns access tokens.
    """
    endpoint = f"{url.rstrip('/')}/auth/v1/token?grant_type=password"
    headers = {
        "apikey": key,
        "Content-Type": "application/json"
    }
    payload = {
        "email": email,
        "password": password
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=7)
        res_data = response.json()
        if response.status_code == 200:
            return True, res_data
        else:
            err_msg = res_data.get('error_description') or res_data.get('msg') or res_data.get('error', {}).get('message') or response.text
            return False, err_msg
    except Exception as e:
        return False, str(e)

def get_user_profile(url, key, user_id):
    """
    Fetches the user's profile information from public.user_profiles.
    """
    endpoint = f"{url.rstrip('/')}/rest/v1/user_profiles?id=eq.{user_id}&select=*"
    try:
        response = requests.get(endpoint, headers=get_headers(key), timeout=7)
        response.raise_for_status()
        profiles = response.json()
        if profiles:
            return profiles[0]
        return None
    except Exception as e:
        print(f"Error fetching user profile {user_id}: {e}")
        return None

def get_pending_profiles(url, key):
    """
    Retrieves all user profiles that are pending approval.
    """
    endpoint = f"{url.rstrip('/')}/rest/v1/user_profiles?is_approved=eq.false&select=*&order=created_at.asc"
    try:
        response = requests.get(endpoint, headers=get_headers(key), timeout=7)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching pending profiles: {e}")
        return []

def approve_profile(url, key, user_id):
    """
    Approves a user account by setting is_approved to True in public.user_profiles.
    """
    endpoint = f"{url.rstrip('/')}/rest/v1/user_profiles?id=eq.{user_id}"
    payload = {"is_approved": True}
    try:
        response = requests.patch(endpoint, headers=get_headers(key), json=payload, timeout=7)
        response.raise_for_status()
        return True, "User approved successfully!"
    except Exception as e:
        return False, str(e)
