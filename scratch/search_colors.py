import re

with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

color_patterns = [
    r'color:\s*(?:white|#fff|#ffffff)\b',
    r'color\s*=\s*["\'](?:white|#fff|#ffffff)["\']'
]

print("Searching for hardcoded light/white colors...")
for idx, line in enumerate(lines, 1):
    for pattern in color_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            print(f"Line {idx}: {line.strip()}")
            break
