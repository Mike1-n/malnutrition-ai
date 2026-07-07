import re

with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

def search_term(pattern):
    print(f"Searching for: {pattern}")
    compiled = re.compile(pattern, re.IGNORECASE)
    for idx, line in enumerate(lines):
        if compiled.search(line):
            print(f"{idx+1}: {line.strip()}")

search_term(r"read_csv|to_csv")
search_term(r"db\.")
search_term(r"get_all_assessments|save_assessment|get_assessments_by_subject|signup_user|signin_user")
