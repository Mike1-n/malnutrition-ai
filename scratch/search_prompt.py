import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Search for JSON or prompt rendering
matches = re.findall(r'.{0,40}(?:json|prompt).{0,40}', content, re.IGNORECASE)
for match in matches[:30]:
    print(match)
