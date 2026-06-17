with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(lines, 1):
    if "unaffected" in line.lower():
        print(f"Line {idx}: {line.strip()}")
