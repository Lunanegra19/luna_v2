with open('c:/Users/Usuario/Downloads/feature_selection_e.py', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'X_lagged = self.lag_disc.transform' in l:
        print('X_lagged:', i)
    elif 'return {' in l:
        print('return:', i)
    elif 'if __name__ == "__main__":' in l:
        print('main:', i)
