import re

with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

color_patterns = [
    r'color:\s*(?:white|#fff|#ffffff)\b',
    r'color\s*=\s*["\'](?:white|#fff|#ffffff)["\']'
]

results = []
for idx, line in enumerate(lines, 1):
    for pattern in color_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            # Strip non-ascii characters for terminal safety
            sanitized = line.encode('ascii', errors='ignore').decode('ascii').strip()
            results.append(f"Line {idx}: {sanitized}")
            break

with open("scratch/color_results.txt", "w") as f:
    f.write("\n".join(results))

print(f"Done! Found {len(results)} matches.")
