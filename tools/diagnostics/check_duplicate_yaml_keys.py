import re
from collections import defaultdict

def find_duplicates(filepath):
    duplicates = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # We will track keys at different indentation levels
    # Since settings.yaml is relatively simple, we can just look for key: patterns
    # and their indent level.
    
    current_path = []
    seen_paths = set()
    
    for line_num, line in enumerate(lines, 1):
        if line.strip().startswith('#') or not line.strip():
            continue
            
        match = re.match(r'^(\s*)([\w-]+)\s*:', line)
        if match:
            indent = len(match.group(1))
            key = match.group(2)
            
            # Pop items from current_path until we match the indent
            while current_path and current_path[-1][1] >= indent:
                current_path.pop()
                
            full_path = '.'.join([p[0] for p in current_path] + [key])
            current_path.append((key, indent))
            
            if full_path in seen_paths:
                duplicates.append((line_num, full_path))
            else:
                seen_paths.add(full_path)
                
    return duplicates

dups = find_duplicates('config/settings.yaml')
if dups:
    print("Found duplicates:")
    for line, path in dups:
        print(f"Line {line}: {path}")
else:
    print("No duplicates found.")
