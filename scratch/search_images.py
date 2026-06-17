with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

import re
matches = re.findall(r'.{0,40}(?:image|\.png|\.jpg|\.jpeg|\bst\.img).{0,40}', content, re.IGNORECASE)
print(f"Found {len(matches)} matches:")
for m in matches[:30]:
    print(m)
