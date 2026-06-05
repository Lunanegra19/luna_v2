import os

def find_decay_fraction(root_dir):
    for r, _, fs in os.walk(root_dir):
        # Skip heavy directories
        if any(skip in r.lower() for skip in ['data', '.git', '__pycache__', 'env', 'venv']):
            continue
        for f in fs:
            if f.endswith(('.py', '.yaml', '.json')):
                path = os.path.join(r, f)
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        for i, line in enumerate(file):
                            if 'pt_decay_fraction' in line:
                                print(f'{path}:{i+1}: {line.strip()}')
                except Exception:
                    pass

find_decay_fraction(r'G:\Mi unidad\ia\Luna v1')
