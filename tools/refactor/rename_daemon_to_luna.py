import os
import re

# Refactor script to rename all references of "Luna v2" to "Luna v2"
# in compliance with structured refactoring and print requirements.

TARGET_DIRS = [
    r"g:\Mi unidad\ia\luna_v2\luna",
    r"g:\Mi unidad\ia\luna_v2\scripts",
    r"g:\Mi unidad\ia\luna_v2\tests",
    r"g:\Mi unidad\ia\luna_v2\docs",
    r"g:\Mi unidad\ia\luna_v2\config",
    r"g:\Mi unidad\ia\luna_v2\tools",
    r"C:\Users\Usuario\.gemini\antigravity-ide\brain\fd276a25-382c-463b-a633-6e14e3db0da1"
]

REPLACEMENTS = [
    # Exact case-sensitive strings
    (r"Luna v2", "Luna v2"),
    (r"Luna v2", "Luna v2"),
    (r"luna-v2", "luna-v2"),
    (r"luna_v2", "luna_v2"),
    
    # Specific SOP tags and prints
    (r"\[V10-CALIB\]", "[LUNA-V2-CALIB]"),
    (r"\[V10-EMBARGO\]", "[LUNA-V2-EMBARGO]"),
    (r"\[V10-REGULARIZATION\]", "[LUNA-V2-REGULARIZATION]"),
    (r"\[V10-CONFIG\]", "[LUNA-V2-CONFIG]"),
    (r"\[CRITICAL-V10\]", "[CRITICAL-LUNA-V2]"),
    
    # Capitalized names in comments
    (r"SOP Luna v2", "SOP Luna v2"),
    (r"Luna v2", "Luna v2"),
    (r"Luna v2 Live", "Luna v2 Live"),
    (r"Risk Monitor Luna v2", "Risk Monitor Luna v2"),
    (r"Luna v2", "Luna v2"),
    
    # Generic "Luna v2" in lower cases
    (r"luna v2", "luna v2"),
]

def apply_rename():
    total_files_modified = 0
    total_replacements = 0

    print("[LUNA-V2-REFACTOR] Starting Luna v2 to Luna v2 migration...")
    
    for target_dir in TARGET_DIRS:
        if not os.path.exists(target_dir):
            print(f"[WARNING] Directory not found: {target_dir}")
            continue
            
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if not file.endswith((".py", ".md", ".yaml", ".json", ".txt")):
                    continue
                    
                file_path = os.path.join(root, file)
                
                # Read content
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, "r", encoding="latin-1") as f:
                            content = f.read()
                    except Exception as e:
                        print(f"[ERROR] Failed to read {file_path}: {e}")
                        continue
                except Exception as e:
                    print(f"[ERROR] Failed to read {file_path}: {e}")
                    continue

                # Apply replacements
                modified_content = content
                file_replaced_count = 0
                
                for pattern, replacement in REPLACEMENTS:
                    matches = re.findall(pattern, modified_content)
                    if matches:
                        modified_content = re.sub(pattern, replacement, modified_content)
                        file_replaced_count += len(matches)
                        
                if file_replaced_count > 0:
                    try:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(modified_content)
                        print(f"[REPLACED] {file_replaced_count} occurrences in: {file_path}")
                        total_files_modified += 1
                        total_replacements += file_replaced_count
                    except Exception as e:
                        print(f"[ERROR] Failed to write {file_path}: {e}")

    print(f"[LUNA-V2-REFACTOR-COMPLETE] Modified {total_files_modified} files with {total_replacements} total replacements.")

if __name__ == "__main__":
    apply_rename()
