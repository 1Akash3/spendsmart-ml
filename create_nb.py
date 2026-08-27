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
add_md("This notebook serves **only** as the execution orchestrator. All implementation logic exists in `src/` to ensure full reproducibility.\n\n### Execution Modes:\n- `smoke`: Rapid CPU-safe pipeline software verification. *Never used for paper claims.*\n- `development`: GPU hyperparameter sweeps & ablations. Uses massive datasets.\n- `final`: Immutable evaluation of the final chosen architecture. Triggering this checks `reports/final_model_manifest.json`.")

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
MODE = "smoke" # Options: smoke, development, final
print(f"🔥 CURRENT EXECUTION MODE: {MODE.upper()}")
''')

add_md("## [Optional] Final Model Locking")
add_md("If you are ready for the `final` execution, run the cell below to freeze your configuration. If you edit hyperparameters after this, the `final` test runner will crash!")
add_code('''if MODE == "final":
    from src.locked_test_guard import LockedTestGuard
    # Freeze the architecture before the run
    LockedTestGuard.create_manifest({
        "architecture": "PATFormer",
        "sequence_length": 64,
        "embedding_dimension": 96,
        "layers": 3,
        "dropout": 0.15,
        "learning_rate": 0.001,
        "optimizer": "AdamW",
        "batch_size": 16,
        "preprocessing_version": "v1.0",
        "dataset_hash": "mocked_data_hash_for_now"
    })
''')

add_md("## 1. Run Data Preprocessing & Splitting (Job 1)")
add_code('''!python src/run_data_pipeline.py --mode {MODE}''')

add_md("## 2. Run Baselines & Categorization (Job 2 & 3)")
add_code('''!python src/run_baselines.py --mode {MODE}''')

add_md("## 3. Run Neural Experiments (PATFormer & Ablations) (Job 4-9)")
add_code('''!python src/run_neural_experiments.py --mode {MODE}''')

add_md("## 4. Render Final Tables from Persisted Artifacts")
add_code('''
reports_dir = Path(f"reports/results/{MODE}")
if not reports_dir.exists():
    print(f"No tables generated yet for mode: {MODE}!")
else:
    for table_file in sorted(reports_dir.glob("*.csv")):
        df = pd.read_csv(table_file)
        display(Markdown(f"### {table_file.stem} ({MODE.upper()})"))
        display(df)
''')

output_file = "F:/Full Stack/spendsmart-ml/notebooks/SpendSmart_Research_Complete.ipynb"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully.")
