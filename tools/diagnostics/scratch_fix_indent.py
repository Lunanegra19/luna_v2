import sys

with open('g:/Mi unidad/ia/luna_v2/luna/features/feature_selection_e.py', encoding='utf-8') as f:
    lines = f.readlines()

idx = -1
for i, l in enumerate(lines):
    if l.startswith('if __name__ == "__main__":'):
        idx = i
        break

if idx != -1:
    for i in range(idx + 1, len(lines)):
        if lines[i].strip():
            lines[i] = '    ' + lines[i].lstrip()

with open('g:/Mi unidad/ia/luna_v2/luna/features/feature_selection_e.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
