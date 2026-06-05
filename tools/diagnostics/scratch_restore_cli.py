with open('c:/Users/Usuario/Downloads/feature_selection_e.py', encoding='utf-8', errors='replace') as f:
    lines_orig = f.readlines()

idx = -1
for i, l in enumerate(lines_orig):
    if l.startswith('if __name__ == "__main__":'):
        idx = i
        break

with open('g:/Mi unidad/ia/luna_v2/luna/features/feature_selection_e.py', encoding='utf-8') as f:
    lines_current = f.readlines()

idx_current = -1
for i, l in enumerate(lines_current):
    if l.startswith('if __name__ == "__main__":'):
        idx_current = i
        break

if idx != -1 and idx_current != -1:
    lines_current = lines_current[:idx_current] + lines_orig[idx:]
    with open('g:/Mi unidad/ia/luna_v2/luna/features/feature_selection_e.py', 'w', encoding='utf-8') as f:
        f.writelines(lines_current)
