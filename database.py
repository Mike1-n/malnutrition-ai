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
    Saves an assessment record to the Supabase database.
    
    url: The Supabase project URL.
    key: The Supabase anon/service_role key.
    data: dict containing the assessment fields.
    
    Returns (success: bool, message: str)
    """
    endpoint = f"{url.rstrip('/')}/rest/v1/assessments"
    try:
        response = requests.post(endpoint, headers=get_headers(key), json=data)
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
        response = requests.get(endpoint, headers=get_headers(key))
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
        response = requests.get(endpoint, headers=get_headers(key))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching assessments for subject {subject_id}: {e}")
        return []
