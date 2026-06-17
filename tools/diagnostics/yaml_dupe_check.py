import sys
import re
from pathlib import Path

def check_duplicate_yaml_keys(filepath):
    duplicates = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_path = []
        seen_paths = set()
        
        for line_num, line in enumerate(lines, 1):
            if line.strip().startswith('#') or not line.strip() or line.strip().startswith('-'):
                continue
                
            match = re.match(r'^(\s*)([\w-]+)\s*:', line)
            if match:
                indent = len(match.group(1))
                key = match.group(2)
                
                while current_path and current_path[-1][1] >= indent:
                    current_path.pop()
                    
                full_path = '.'.join([p[0] for p in current_path] + [key])
                current_path.append((key, indent))
                
                skip_blocks = ['wfb.windows', 'dashboard_healthcheck.endpoints']
                if any(full_path.startswith(b) for b in skip_blocks):
                    continue
                    
                if full_path in seen_paths:
                    duplicates.append((line_num, full_path))
                else:
                    seen_paths.add(full_path)
    except Exception as e:
        print(f"Error checking YAML keys: {e}")
        return False

    if duplicates:
        print("\n[PRE-FLIGHT] ?? PIPELINE BLOQUEADO — Se detectaron parámetros duplicados en settings.yaml:")
        for line, path in duplicates:
            print(f"  Line {line}: {path}")
        print("[PRE-FLIGHT] La política institucional 'No-Fallback' prohíbe llaves duplicadas, ya que el parser ignoraría una de ellas.\n")
        return False
    return True

if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'config/settings.yaml'
    if not check_duplicate_yaml_keys(filepath):
        sys.exit(1)
