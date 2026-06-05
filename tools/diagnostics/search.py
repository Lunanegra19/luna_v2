import os
import sys

def search(d):
    for r, _, fs in os.walk(d):
        for f in fs:
            if f.endswith('.py') or f.endswith('.yaml'):
                path = os.path.join(r, f)
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        if 'LUNA_SMOKE_TEST' in file.read():
                            print(f'FOUND IN: {path}')
                except Exception:
                    pass

search('.')
