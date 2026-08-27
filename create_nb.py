import json
import os

notebook = {
  "cells": [],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}

def add_md(text):
    lines = text.split('\n')
    notebook["cells"].append({
      "cell_type": "markdown",
      "metadata": {},
      "source": [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]
    })

def add_code(text):
    lines = text.split('\n')
    notebook["cells"].append({
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]
    })

add_md("# 💸 SpendSmart-ML — Final Empirical Research Orchestrator\n## Google Colab GPU Edition")
add_md("This notebook serves **only** as the execution orchestrator. All implementation logic exists in `src/` to ensure full reproducibility.\n\n### Execution Modes:\n- `SMOKE_TEST`: Rapid CPU-safe pipeline verification.\n- `DEVELOPMENT`: GPU hyperparameter sweeps & ablations.\n- `FINAL_LOCKED_TEST`: Immutable evaluation of the final chosen architecture.")

add_code('''import os
import sys
import subprocess

REPO_URL = 'https://github.com/1Akash3/spendsmart-ml.git'
PROJECT_DIR = 'spendsmart-ml'

if REPO_URL and not os.path.isdir(PROJECT_DIR):
    subprocess.run(["git", "clone", REPO_URL], check=True)

if not os.path.isdir(PROJECT_DIR):
    PROJECT_DIR = '.'

os.chdir(PROJECT_DIR)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
print('✅ Working dir:', os.getcwd())''')

add_code('''!pip -q install -r requirements.txt
!pip -q install datasets pdfplumber huggingface_hub torch''')

add_code('''import pandas as pd
import json
import glob
from pathlib import Path
from IPython.display import display, Markdown

# Mode Selection
MODE = "SMOKE_TEST" # Options: SMOKE_TEST, DEVELOPMENT, FINAL_LOCKED_TEST
print(f"🔥 CURRENT EXECUTION MODE: {MODE}")

if MODE == "FINAL_LOCKED_TEST":
    print("🔒 LOCKED TEST MODE INITIATED: Hyperparameters and Architecture Frozen.")
''')

add_md("## 1. Run Data Preprocessing & Splitting (Job 1)")
add_code('''!python src/run_data_pipeline.py --mode {MODE}''')

add_md("## 2. Run Baselines & Categorization (Job 2 & 3)")
add_code('''!python src/run_baselines.py --mode {MODE}''')

add_md("## 3. Run Neural Experiments (PATFormer & Ablations) (Job 4-9)")
add_code('''!python src/run_neural_experiments.py --mode {MODE}''')

add_md("## 4. Render Final Tables from Artifacts")
add_code('''
reports_dir = Path("reports/tables")
if not reports_dir.exists():
    print("No tables generated yet!")
else:
    for table_file in sorted(reports_dir.glob("*.csv")):
        df = pd.read_csv(table_file)
        display(Markdown(f"### {table_file.stem}"))
        display(df)
''')

output_file = "F:/Full Stack/spendsmart-ml/notebooks/SpendSmart_Research_Complete.ipynb"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully.")
