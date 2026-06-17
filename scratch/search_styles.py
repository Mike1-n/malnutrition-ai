import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find all style="..." matches
styles = re.findall(r'style=["\']([^"\']+)["\']', content)

print(f"Found {len(styles)} style attributes. Filtering for color/background specifications:")
for idx, style in enumerate(styles, 1):
    if 'color' in style or 'background' in style:
        print(f"Style {idx}: {style}")
