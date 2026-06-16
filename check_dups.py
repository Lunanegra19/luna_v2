import yaml

def get_leaves(d, current_path=''):
    leaves = []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                if isinstance(v, dict):
                    leaves.extend(get_leaves(v, current_path + '.' + k if current_path else k))
                else:
                    leaves.append((k, current_path + '.' + k if current_path else k))
            else:
                leaves.append((k, current_path + '.' + k if current_path else k))
    return leaves

d = yaml.safe_load(open('config/settings.yaml'))
leaves = get_leaves(d)

counts = {}
paths = {}
for k, path in leaves:
    counts[k] = counts.get(k, 0) + 1
    paths.setdefault(k, []).append(path)

for k, c in counts.items():
    if c > 1:
        print(f"Dup: {k} -> {paths[k]}")
