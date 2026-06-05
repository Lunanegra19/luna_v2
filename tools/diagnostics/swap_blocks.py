import sys

with open('luna/features/feature_pipeline.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_kmeans = -1
start_mcs = -1
end_mcs = -1

for i, line in enumerate(lines):
    if 'if "KMeans_Tribe_ID" not in df.columns:' in line:
        start_kmeans = i - 1
    elif 'if "Master_Causal_Signal" not in df.columns:' in line:
        start_mcs = i - 1
    elif 'if "vix_slope_7d" not in df.columns:' in line:
        end_mcs = i - 2
        
if start_kmeans != -1 and start_mcs != -1 and end_mcs != -1:
    print('Found blocks:')
    print(f'KMeans: {start_kmeans} to {start_mcs}')
    print(f'MCS: {start_mcs} to {end_mcs}')
    
    pre_block = lines[:start_kmeans]
    kmeans_block = lines[start_kmeans:start_mcs]
    mcs_block = lines[start_mcs:end_mcs]
    post_block = lines[end_mcs:]
    
    new_lines = pre_block + mcs_block + ['\n'] + kmeans_block + post_block
    
    with open('luna/features/feature_pipeline.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print('Swapped successfully')
else:
    print('Failed to find blocks', start_kmeans, start_mcs, end_mcs)
