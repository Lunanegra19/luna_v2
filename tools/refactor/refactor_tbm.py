import os
import glob

def refactor_tbm():
    target_dir = r"G:\Mi unidad\ia\luna_v2"
    py_files = glob.glob(os.path.join(target_dir, "**", "*.py"), recursive=True)
    
    replacements = [
        ("luna.labeling.triple_barrier", "luna.features.tbm")
    ]
    
    modified_count = 0
    for file_path in py_files:
        if "refactor_tbm.py" in file_path:
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            encoding_used = "utf-8"
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
            encoding_used = "latin-1"
            
        new_content = content
        for old, new in replacements:
            new_content = new_content.replace(old, new)
            
        if new_content != content:
            with open(file_path, "w", encoding=encoding_used) as f:
                f.write(new_content)
            modified_count += 1
            print(f"Modified: {file_path}")
            
    print(f"Total files modified: {modified_count}")

if __name__ == "__main__":
    refactor_tbm()
