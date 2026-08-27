import os
import subprocess
import sys
import json

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def verify():
    print("=== PHASE 1: REPOSITORY VERIFICATION ===")
    
    branch, _, _ = run_cmd("git branch --show-current")
    commit, _, _ = run_cmd("git log -1 --oneline")
    status, _, _ = run_cmd("git status --short")
    
    print(f"1. Git Branch: {branch}")
    print(f"2. Git Commit: {commit}")
    print(f"3. Working Tree: {'Clean' if not status else 'DIRTY'}")
    
    print("4. Running Test Suite...")
    _, stderr, rc = run_cmd("pytest tests/")
    if rc == 0:
        print("   -> Test suite PASS")
    else:
        print("   -> Test suite FAIL")
        print(stderr)
        
    print("5. Running Compile Checks...")
    _, err, rc2 = run_cmd("python -m compileall src")
    if rc2 == 0:
        print("   -> Compile checks PASS")
    else:
        print("   -> Compile checks FAIL")
        
    print("6. Verifying Notebook Mock Status...")
    nb_path = "notebooks/SpendSmart_Research_Complete.ipynb"
    if os.path.exists(nb_path):
        with open(nb_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "pd.DataFrame({" in content and "0.77" in content:
                print("   -> WARNING: Mocked DataFrames detected in Notebook! Must replace with dynamic loading.")
            else:
                print("   -> Notebook is clean of static mocks.")
    else:
        print("   -> Notebook not found!")

if __name__ == "__main__":
    verify()
