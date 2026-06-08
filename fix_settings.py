import re

with open('config/settings.yaml', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'optuna_metric:\s*brier', r'optuna_metric: dsr', content)

content = re.sub(r'reg_alpha_max:\s*0\.2', r'reg_alpha_max: 50.0', content)
content = re.sub(r'reg_lambda_max:\s*1\.0', r'reg_lambda_max: 100.0', content)
content = re.sub(r'learning_rate_max:\s*0\.15', r'learning_rate_max: 0.05', content)

with open('config/settings.yaml', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done!')
