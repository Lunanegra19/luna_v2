import joblib
import glob
files = glob.glob('/root/luna_v2/data/models/prod/seed*/hmm_regime.pkl')
if files:
    d = joblib.load(files[0])
    print(d.get('features'))
else:
    print("No file found")
