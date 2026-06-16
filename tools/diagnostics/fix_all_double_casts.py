import os
import re

def fix_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read()

    original = content
    content = re.sub(r'float\(\s*int\(\s*(_cfg[^\)]+)\)\s*\)', r'float(\1)', content)
    content = re.sub(r'int\(\s*int\(\s*(_cfg[^\)]+)\)\s*\)', r'int(\1)', content)
    content = re.sub(r'float\(\s*float\(\s*(_cfg[^\)]+)\)\s*\)', r'float(\1)', content)
    content = re.sub(r'bool\(\s*int\(\s*(_cfg[^\)]+)\)\s*\)', r'bool(\1)', content)
    content = re.sub(r'bool\(\s*bool\(\s*(_cfg[^\)]+)\)\s*\)', r'bool(\1)', content)

    if content != original:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception:
            with open(filepath, 'w', encoding='latin-1') as f:
                f.write(content)
        print(f"Fixed double casts in {filepath}")
        return True
    return False

luna_dir = 'luna'
fixed_count = 0
for root, _, files in os.walk(luna_dir):
    for file in files:
        if file.endswith('.py'):
            if fix_file(os.path.join(root, file)):
                fixed_count += 1

print(f"Fixed {fixed_count} files.")
