import codecs

with codecs.open('g:/Mi unidad/ia/luna_v2/luna/features/feature_selection_e.py', 'r', encoding='latin-1', errors='ignore') as f:
    text = f.read()

with codecs.open('g:/Mi unidad/ia/luna_v2/luna/features/feature_selection_e.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('File converted to UTF-8.')
