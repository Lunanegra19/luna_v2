"""Fix CHK-09 via line number injection - add in_soh_fn logic"""

path = '/root/luna_v2/dashboard/server.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the chk09 function and fix it
# We know chk09 is at line 2442, in_options is at 2447, and the for loop is at 2448
# We need to:
# 1. Add "in_soh_fn = False" after "in_options = False" 
# 2. Add in_soh_fn detection at start of loop
# 3. Add skip condition

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    # Find the in_options = False inside chk09
    if 'def chk09' in line:
        new_lines.append(line)
        i += 1
        # Look for "in_options = False" in the next 10 lines
        j = i
        while j < i + 15 and j < len(lines):
            if lines[j].strip() == 'in_options = False':
                new_lines.append(lines[j])
                # Insert in_soh_fn variable
                indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
                new_lines.append(f'{indent}in_soh_fn = False  # [FIX-CHK09] Excluir bloque run_sop_health_checks\n')
                j += 1
                # Now find "for i, line in enumerate(lines):" and add in_soh_fn checks
                while j < i + 25 and j < len(lines):
                    l = lines[j]
                    if 'for i, line in enumerate(lines):' in l:
                        new_lines.append(l)
                        j += 1
                        # Add in_soh_fn detection after the for loop
                        ls = lines[j]
                        indent2 = ls[:len(ls) - len(ls.lstrip())]
                        new_lines.append(f'{indent2}if "def run_sop_health_checks" in s: in_soh_fn = True\n')
                        new_lines.append(f'{indent2}if in_soh_fn and "class DashboardHTTPHandler" in s: in_soh_fn = False\n')
                        new_lines.append(f'{indent2}if in_soh_fn: continue  # skip SOH fn body - contiene literales de codigo\n')
                        break
                    else:
                        new_lines.append(l)
                        j += 1
                # Continue from j
                i = j
                break
            else:
                new_lines.append(lines[j])
                j += 1
        else:
            i = j
        continue
    new_lines.append(line)
    i += 1

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("[CHK-09-FIX] in_soh_fn logic añadida")

# Verify syntax
import subprocess
r = subprocess.run(['/root/miniconda3/envs/luna_env/bin/python', '-m', 'py_compile', path],
                  capture_output=True, text=True)
print(f"[CHK-09-FIX] Sintaxis: {'OK' if r.returncode == 0 else 'ERROR: ' + r.stderr}")

# Check the fix is there
with open(path, 'r') as f:
    c = f.read()
print(f"[CHK-09-FIX] in_soh_fn presente: {'in_soh_fn' in c}")
