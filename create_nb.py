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

add_md("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1Akash3/spendsmart-ml/blob/main/notebooks/SpendSmart_Research_Complete.ipynb)")
add_md("# 💸 SpendSmart-ML — Research-Grade Personalized Spending Intelligence\n## Complete Pipeline with Algorithm Comparison & Real-World Validation\nExplain: research paper pipeline, CPU-only, benchmarked against baselines")
add_md("## 1 · Environment Setup")
add_code('''import os, sys
REPO_URL = 'https://github.com/1Akash3/spendsmart-ml.git'
PROJECT_DIR = 'spendsmart-ml'
if REPO_URL and not os.path.isdir(PROJECT_DIR):
    !git clone -q $REPO_URL
if not os.path.isdir(PROJECT_DIR):
    PROJECT_DIR = '.' if os.path.isfile('config.py') else PROJECT_DIR
assert os.path.isdir(PROJECT_DIR), 'Upload the spendsmart-ml folder or set REPO_URL.'
os.chdir(PROJECT_DIR)
print('Working dir:', os.getcwd())''')
add_code('''!pip -q install -r requirements.txt
!pip -q install datasets pdfplumber huggingface_hub
import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
print('Dependencies installed, paths configured.')''')
add_md("### 🔑 API Credentials (Colab Secrets)\n\nThis notebook uses **Colab Secrets** for API keys — nothing is hardcoded.\n\n1. Click the **key icon (🔑)** in the left sidebar → **Secrets**\n2. Add: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `HF_TOKEN`\n3. Toggle **Notebook access** on for each")
add_code('''from google.colab import userdata
import os
try:
    os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
    print('✅ Kaggle credentials loaded.')
except Exception as e:
    print(f'⚠️ Kaggle credentials not found in Secrets: {e}')
    print('   You can still run synthetic mode.')
try:
    from huggingface_hub import login
    login(token=userdata.get('HF_TOKEN'))
    print('✅ HuggingFace token loaded.')
except Exception as e:
    print(f'⚠️ HF token not found: {e}')''')
add_md("### 💾 Google Drive Persistence")
add_code('''try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_OUT = '/content/drive/MyDrive/spendsmart-ml-outputs'
    os.makedirs(DRIVE_OUT, exist_ok=True)
    print('Will copy artifacts/reports to:', DRIVE_OUT)
except Exception:
    DRIVE_OUT = None
    print('Drive not available — results stay in the VM.')''')
add_md("---\n## 2 · Load Real Transaction Data")
add_code('''import numpy as np
import pandas as pd
from data_sources import load_all_real_transactions, load_all_labeled_descriptions

try:
    txns = load_all_real_transactions()
    labeled = load_all_labeled_descriptions()
    print(f'✅ Loaded {len(txns):,} transactions, {txns.user_id.nunique()} users')
    print(f'✅ Loaded {len(labeled):,} labeled descriptions for categorizer training')
    DATA_SOURCE = 'combined'
except Exception as e:
    print(f'⚠️ Real data load failed: {e}')
    print('Falling back to synthetic data...')
    from synth import generate_transactions
    txns = generate_transactions(n_users=500, months=18, seed=42)
    labeled = txns[['description', 'category']].dropna()
    DATA_SOURCE = 'synthetic'

print(f'\nData source: {DATA_SOURCE}')
print(f'Categories: {sorted(txns.category.unique())}')
txns.head()''')
add_md("---\n## 3 · 📊 Algorithm Comparison: Transaction Categorizer\n\nWe compare **three approaches** to classify transaction descriptions into spending categories:\n1. **TF-IDF Baseline** — word + char n-gram features → logistic regression\n2. **Hybrid Model** — TF-IDF + sentence-transformer embeddings → logistic regression\n3. **Naive Majority Baseline** — always predicts the most common category\n\nThis is the core comparison for the research paper.")
add_code('''import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from categorizer import TransactionCategorizer, HybridTransactionCategorizer
from config import RANDOM_SEED

# Prepare data
cdf = labeled.dropna()
if len(cdf) > 200_000:
    cdf = cdf.sample(200_000, random_state=RANDOM_SEED)
desc = cdf['description'].values
labels = cdf['category'].values
Xtr, Xte, ytr, yte = train_test_split(desc, labels, test_size=0.2,
                                      random_state=RANDOM_SEED, stratify=labels)
print(f'Train: {len(Xtr):,}  |  Test: {len(Xte):,}  |  Classes: {len(set(yte))}')

results = {}

# --- Naive Majority Baseline ---
from collections import Counter
majority_class = Counter(ytr).most_common(1)[0][0]
y_naive = [majority_class] * len(yte)
results['Naive Majority'] = {
    'accuracy': accuracy_score(yte, y_naive),
    'macro_f1': f1_score(yte, y_naive, average='macro', zero_division=0),
    'weighted_f1': f1_score(yte, y_naive, average='weighted', zero_division=0),
    'train_time': 0.0,
}
print(f"\n{'='*60}")
print(f'Naive Majority Baseline: accuracy={results["Naive Majority"]["accuracy"]:.4f}')

# --- TF-IDF Baseline ---
t0 = time.time()
tfidf_cat = TransactionCategorizer().fit(Xtr, ytr)
tfidf_time = time.time() - t0
tfidf_eval = tfidf_cat.evaluate(Xte, yte)
results['TF-IDF (Baseline)'] = {
    'accuracy': tfidf_eval['accuracy'],
    'macro_f1': tfidf_eval['macro_f1'],
    'weighted_f1': tfidf_eval['weighted_f1'],
    'train_time': round(tfidf_time, 2),
    'report': tfidf_eval['report'],
}
print(f"\n{'='*60}")
print(f"TF-IDF Baseline: accuracy={tfidf_eval['accuracy']:.4f}  macro-F1={tfidf_eval['macro_f1']:.4f}  ({tfidf_time:.1f}s)")

# --- Hybrid (TF-IDF + Transformer) ---
try:
    t0 = time.time()
    hybrid_cat = HybridTransactionCategorizer().fit(Xtr, ytr)
    hybrid_time = time.time() - t0
    hybrid_eval = hybrid_cat.evaluate(Xte, yte)
    results['Hybrid (TF-IDF + Transformer)'] = {
        'accuracy': hybrid_eval['accuracy'],
        'macro_f1': hybrid_eval['macro_f1'],
        'weighted_f1': hybrid_eval['weighted_f1'],
        'train_time': round(hybrid_time, 2),
        'report': hybrid_eval['report'],
    }
    print(f"\n{'='*60}")
    print(f"Hybrid Model: accuracy={hybrid_eval['accuracy']:.4f}  macro-F1={hybrid_eval['macro_f1']:.4f}  ({hybrid_time:.1f}s)")
except Exception as e:
    print(f'⚠️ Hybrid model failed: {e}')
    hybrid_cat = None

print(f"\n{'='*60}")
print('COMPARISON SUMMARY:')
for name, r in results.items():
    print(f"  {name:35s}  acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}  time={r['train_time']:.1f}s")''')
add_md("### Categorizer Comparison Chart")
add_code('''import matplotlib.pyplot as plt
import numpy as np

names = list(results.keys())
acc = [results[n]['accuracy'] for n in names]
f1 = [results[n]['macro_f1'] for n in names]

x = np.arange(len(names))
w = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
bar1 = ax.bar(x - w/2, acc, w, label='Accuracy', color='#2196F3')
bar2 = ax.bar(x + w/2, f1, w, label='Macro-F1', color='#FF9800')
ax.set_ylabel('Score')
ax.set_title('Transaction Categorizer — Algorithm Comparison')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha='right')
ax.set_ylim(0, 1.1)
ax.legend()
for bar in bar1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bar2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('reports/categorizer_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: reports/categorizer_comparison.png')''')
add_md("### Per-Class Performance (Best Model)")
add_code('''import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from config import CATEGORY_LABELS

# Use the best model
if hybrid_cat is not None:
    best_name = 'Hybrid (TF-IDF + Transformer)'
    best_model = hybrid_cat
else:
    best_name = 'TF-IDF (Baseline)'
    best_model = tfidf_cat

print(f'Best model: {best_name}')
print(f"\nPer-class classification report:")
y_pred = best_model.predict(Xte)
print(classification_report(yte, y_pred, zero_division=0))

# Confusion matrix
class_names = sorted(set(yte))
cm = confusion_matrix(yte, y_pred, labels=class_names)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[CATEGORY_LABELS.get(c, c) for c in class_names],
            yticklabels=[CATEGORY_LABELS.get(c, c) for c in class_names], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix — {best_name}')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: reports/confusion_matrix.png')''')
add_md("---\n## 4 · Monthly Panel, User Profiles & Segmentation")
add_code('''from features import build_monthly_panel, build_user_profiles, PROFILE_FEATURE_COLS
from segmentation import UserSegmenter
import config

panel = build_monthly_panel(txns)
panel['income'] = panel['income'].fillna(0.0)
profiles = build_user_profiles(panel).replace([np.inf, -np.inf], np.nan).fillna(0.0)

seg = UserSegmenter().fit(profiles)
static = seg.assign(profiles)[PROFILE_FEATURE_COLS + ['cohort']].reset_index()

print(f'Panel: {len(panel):,} rows, {panel.user_id.nunique()} users')
print(f'Profiles: {len(profiles)} users')
print(f'Segmentation: {seg.n_cohorts} cohorts, silhouette={seg.silhouette_:.4f}')
print(f'\nCohort sizes:')
print(static['cohort'].value_counts().sort_index())''')
add_md("### Cohort Visualization")
add_code('''from sklearn.decomposition import PCA

X = profiles[PROFILE_FEATURE_COLS].values
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)
cohorts = seg.predict(profiles)

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X2[:, 0], X2[:, 1], c=cohorts, cmap='Set2', alpha=0.5, s=15)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.set_title(f'User Segments (K={seg.n_cohorts}, silhouette={seg.silhouette_:.3f})')
plt.colorbar(scatter, label='Cohort')
plt.tight_layout()
plt.savefig('reports/cohort_pca.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: reports/cohort_pca.png')''')
add_md("---\n## 5 · 📈 Personalized Forecaster — Backtest & Comparison")
add_code('''from evaluate import backtest_forecaster, evaluate_overspend_detector
from config import EXPENSE_CATEGORIES

fc_metrics = backtest_forecaster(panel, static)

print(f"Forecaster Backtest Results:")
print(f"  Model MAE:         {fc_metrics['model_mae']}")
print(f"  Naive MAE:         {fc_metrics['naive_mae']}")
print(f"  Rolling Mean MAE:  {fc_metrics['rolling_mean_mae']}")
print(f"  Model WAPE:        {fc_metrics.get('model_wape')}")
print(f"  Model sMAPE:       {fc_metrics.get('model_smape')}")
print(f"  Skill vs Naive:    {fc_metrics['skill_vs_naive']:+.1%}")
print(f"  Train rows: {fc_metrics['n_train_rows']:,}  Test rows: {fc_metrics['n_test_rows']:,}")''')
add_md("### Per-Category Forecast Error")
add_code('''pc = fc_metrics['per_category_mae']
cats = list(pc.keys())
model_mae = [pc[c]['model_mae'] for c in cats]
naive_mae = [pc[c]['naive_mae'] for c in cats]

x = np.arange(len(cats))
w = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w/2, model_mae, w, label='Personalized Model', color='#4CAF50')
ax.bar(x + w/2, naive_mae, w, label='Naive (Last Month)', color='#F44336')
ax.set_xticks(x)
ax.set_xticklabels([CATEGORY_LABELS.get(c, c) for c in cats], rotation=45, ha='right')
ax.set_ylabel('MAE (lower = better)')
ax.set_title('Per-Category Forecast Error: Model vs Naive Baseline')
ax.legend()
plt.tight_layout()
plt.savefig('reports/forecast_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: reports/forecast_comparison.png')''')
add_md("---\n## 6 · Overspend Detection Evaluation")
add_code('''rec_metrics = evaluate_overspend_detector(panel, static)
if 'model' in rec_metrics:
    print('Overspend Detection Results:')
    print(f"  Model     — Precision: {rec_metrics['model']['precision']:.4f}  "
          f"Recall: {rec_metrics['model']['recall']:.4f}  F1: {rec_metrics['model']['f1']:.4f}")
    print(f"  Naive     — Precision: {rec_metrics['naive_baseline']['precision']:.4f}  "
          f"Recall: {rec_metrics['naive_baseline']['recall']:.4f}  F1: {rec_metrics['naive_baseline']['f1']:.4f}")
    print(f"  Positive rate: {rec_metrics['positive_rate']:.4f}")
    print(f"  Test rows: {rec_metrics['n_test_rows']:,}")
else:
    print('Insufficient data for overspend detection evaluation.')''')
add_md("---\n## 7 · Personalized Recommendation Demo")
add_code('''from features import build_forecast_frame, build_serving_frame
from forecaster import PersonalizedForecaster
from recommender import PersonalizedRecommender, RealTimeState

# Fit final forecaster on all data
full_frame = build_forecast_frame(panel, static_features=static)
forecaster = PersonalizedForecaster().fit(full_frame)
serving = build_serving_frame(panel, static_features=static)
forecasts = forecaster.predict_by_user_category(serving)

# Pick a user with enough history
recommender = PersonalizedRecommender()
demo_uid = None
for cand in profiles.index[:600]:
    cpanel = panel[panel['user_id'] == cand]
    if len(cpanel) >= 6 and cand in forecasts:
        demo_uid = cand
        break

if demo_uid is not None:
    upanel = panel[panel['user_id'] == demo_uid]
    cohort = int(static.loc[static['user_id'] == demo_uid, 'cohort'].iloc[0])
    out = recommender.recommend(
        upanel, forecasts[demo_uid],
        cohort_norms_row=seg.cohort_category_norms_.loc[cohort],
        savings_goal_rate=0.25)
    s = out['summary']
    print(f"User {demo_uid} | Cohort {cohort}")
    print(f"  Income: ₹{s['avg_income']:,.0f}")
    print(f"  Projected expense: ₹{s['projected_expense']:,.0f}")
    print(f"  Projected savings rate: {s['projected_savings_rate']*100:.1f}%\n")
    for i, rec in enumerate(out['recommendations'], 1):
        print(f"  {i}. [{rec['kind']}] {rec['title']}")
        print(f"     {rec['detail']}")
        print(f"     Impact: ~₹{rec['monthly_impact']:,.0f}/mo  (confidence {rec['confidence']:.0%})\n")
    if out['plan']:
        print(f"  Savings Plan: {out['plan']['summary']}")
else:
    print('No suitable user found for demo.')''')
add_md("### User Spending History + Forecast")
add_code('''if demo_uid is not None:
    g = upanel.sort_values('month')
    top = g[EXPENSE_CATEGORIES].mean().sort_values(ascending=False).head(5).index.tolist()
    fig, ax = plt.subplots(figsize=(12, 5))
    for c in top:
        ax.plot(g['month'], g[c], marker='o', label=CATEGORY_LABELS.get(c, c))
        ax.scatter([g['month'].max()], [forecasts[demo_uid][c]], marker='*', s=180, zorder=5)
    ax.set_title(f'User {demo_uid}: Top Categories (★ = Next-Month Forecast)')
    ax.set_ylabel('₹ / month')
    ax.legend()
    plt.tight_layout()
    plt.savefig('reports/user_forecast.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('No user data to plot.')''')
add_md("---\n## 8 · Real-Time Optimization Demo")
add_code('''if demo_uid is not None:
    mean = {c: float(upanel[c].mean()) for c in EXPENSE_CATEGORIES}
    std = {c: float(upanel[c].std(ddof=0)) for c in EXPENSE_CATEGORIES}
    rt = RealTimeState(mean, std, income=float(upanel['income'].mean()))
    stream = [
        ('food_dining', mean['food_dining']*0.5, 3),
        ('food_dining', mean['food_dining']*0.6, 6),
        ('shopping', mean['shopping']*0.4, 7),
        ('food_dining', mean['food_dining']*0.7, 9),
    ]
    print('Simulating mid-month transactions...')
    for cat, amt, day in stream:
        for a in rt.update(cat, amt, day):
            print(f"  Day {day:>2}  ⚠️  {a['message']}")
    print('\nProjected month-end (top 5):')
    pe = rt.projected_month_end()
    for c in sorted(pe, key=pe.get, reverse=True)[:5]:
        print(f"  {CATEGORY_LABELS.get(c, c):<20} ₹{pe[c]:,.0f}")
else:
    print('No user data for real-time demo.')''')
add_md("---\n## 9 · 📄 Google Pay PDF Statement Analysis\n\nUpload your Google Pay transaction statement PDF and the model will parse it, extract transactions, categorize them, and show spending insights.")
add_code('''import re

def parse_gpay_pdf(pdf_path):
    """Parse a Google Pay transaction statement PDF into a DataFrame."""
    import pdfplumber
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ''
            for line in text.split('\n'):
                line = line.strip()
                # Match: PaidtoMerchantName ₹Amount
                m = re.match(r'^Paidto(.+?)\s+₹([\d,]+\.?\d*)', line)
                if m:
                    merchant = m.group(1).strip()
                    amount = float(m.group(2).replace(',', ''))
                    rows.append({'type': 'paid', 'description': merchant, 'amount': amount})
                    continue
                # Match: ReceivedfromName ₹Amount
                m = re.match(r'^Receivedfrom(.+?)\s+₹([\d,]+\.?\d*)', line)
                if m:
                    sender = m.group(1).strip()
                    amount = float(m.group(2).replace(',', ''))
                    rows.append({'type': 'received', 'description': sender, 'amount': amount})
    if not rows:
        raise ValueError('No transactions found in PDF. Check format.')
    return pd.DataFrame(rows)

print('Google Pay PDF parser ready.')
print('Upload a GPay PDF in the next cell to analyze it.')''')
add_code('''from google.colab import files
import os

print('Upload your Google Pay PDF statement:')
try:
    uploaded = files.upload()
    pdf_name = list(uploaded.keys())[0]
    print(f'\nProcessing: {pdf_name}')
    gpay_df = parse_gpay_pdf(pdf_name)
    print(f'Extracted {len(gpay_df)} transactions')
    print(f'  Paid: {len(gpay_df[gpay_df.type=="paid"])} transactions, ₹{gpay_df[gpay_df.type=="paid"]["amount"].sum():,.0f}')
    print(f'  Received: {len(gpay_df[gpay_df.type=="received"])} transactions, ₹{gpay_df[gpay_df.type=="received"]["amount"].sum():,.0f}')
    
    # Categorize paid transactions
    paid = gpay_df[gpay_df['type'] == 'paid'].copy()
    if len(paid) > 0:
        cat_model = hybrid_cat if hybrid_cat is not None else tfidf_cat
        paid['predicted_category'] = cat_model.predict(paid['description'].astype(str).tolist())
        if hasattr(cat_model, 'predict_with_confidence'):
            _, conf = cat_model.predict_with_confidence(paid['description'].astype(str).tolist())
            paid['confidence'] = conf
        
        print(f'\n📊 Spending by Category:')
        cat_spend = paid.groupby('predicted_category')['amount'].agg(['sum', 'count'])
        cat_spend.columns = ['Total (₹)', 'Count']
        cat_spend = cat_spend.sort_values('Total (₹)', ascending=False)
        cat_spend.index = [CATEGORY_LABELS.get(c, c) for c in cat_spend.index]
        print(cat_spend.to_string())
        
        # Pie chart
        fig, ax = plt.subplots(figsize=(10, 7))
        cat_spend['Total (₹)'].plot.pie(autopct='%1.1f%%', ax=ax, startangle=90)
        ax.set_ylabel('')
        ax.set_title('Google Pay Spending by Category')
        plt.tight_layout()
        plt.savefig('reports/gpay_spending.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save predictions
        paid.to_csv('gpay_predictions.csv', index=False)
        print(f'\n✅ Predictions saved to gpay_predictions.csv')
    
    gpay_df.head(10)
except Exception as e:
    print(f'No file uploaded or error: {e}')
    print('You can also place a PDF in the project folder and call parse_gpay_pdf("filename.pdf") directly.')''')
add_md("---\n## 10 · (Optional) HuggingFace Dataset Training\n\nTrain on the 4.5M-row MIT-licensed HuggingFace transaction categorization dataset.")
add_code('''try:
    from data_sources import load_hf_transaction_categorization
    real_hf = load_hf_transaction_categorization(sample=100_000)
    Xtr_hf, Xte_hf, ytr_hf, yte_hf = train_test_split(
        real_hf['description'], real_hf['category'], test_size=0.2, random_state=42)
    
    hf_tfidf = TransactionCategorizer().fit(Xtr_hf, ytr_hf)
    hf_eval = hf_tfidf.evaluate(Xte_hf, yte_hf)
    print(f"HuggingFace Dataset — TF-IDF: accuracy={hf_eval['accuracy']:.4f}  F1={hf_eval['macro_f1']:.4f}")
    
    results['HF TF-IDF'] = {
        'accuracy': hf_eval['accuracy'],
        'macro_f1': hf_eval['macro_f1'],
        'weighted_f1': hf_eval['weighted_f1'],
        'train_time': 0,
    }
except Exception as e:
    print(f'HF dataset not available: {e}')''')
add_md("---\n## 11 · Save Artifacts & Final Summary")
add_code('''import json
from config import ARTIFACTS_DIR, REPORTS_DIR

# Save models
tfidf_cat.save()
seg.save()
forecaster.save()
if hybrid_cat is not None:
    hybrid_cat.save()

# Compile all metrics
all_metrics = {
    'data_source': DATA_SOURCE,
    'categorizer_comparison': {k: {kk: vv for kk, vv in v.items() if kk != 'report'}
                                for k, v in results.items()},
    'segmentation': {'n_cohorts': seg.n_cohorts, 'silhouette': round(seg.silhouette_, 4)},
    'forecaster': fc_metrics,
    'overspend_detector': rec_metrics,
}
(REPORTS_DIR / 'research_metrics.json').write_text(json.dumps(all_metrics, indent=2, default=str))

# Copy to Drive
if DRIVE_OUT:
    !cp -r artifacts reports "$DRIVE_OUT"/
    print(f'Copied to Drive: {DRIVE_OUT}')

print(f"\n{'='*74}")
print('  RESEARCH RESULTS SUMMARY'.center(74))
print(f"{'='*74}")
print(f'\n  Data: {DATA_SOURCE}')
print(f'\n  CATEGORIZER COMPARISON:')
for name, r in results.items():
    print(f"    {name:35s}  acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}")
print(f'\n  SEGMENTATION: {seg.n_cohorts} cohorts, silhouette={seg.silhouette_:.4f}')
if 'model_mae' in fc_metrics:
    print(f"\n  FORECASTER:")
    print(f"    Model MAE: {fc_metrics['model_mae']}")
    print(f"    Naive MAE: {fc_metrics['naive_mae']}")
    print(f"    Skill: {fc_metrics['skill_vs_naive']:+.1%}")
if 'model' in rec_metrics:
    print(f"\n  OVERSPEND DETECTOR:")
    print(f"    Model F1: {rec_metrics['model']['f1']}")
    print(f"    Naive F1: {rec_metrics['naive_baseline']['f1']}")
print(f"\n{'='*74}")
print(f'  Full metrics: {REPORTS_DIR / "research_metrics.json"}')
print(f'  Artifacts: {ARTIFACTS_DIR}')
print(f"{'='*74}")''')
add_md("---\nThis notebook produces all artifacts and metrics needed for the SpendSmart-ML research paper. All results are reproducible with seed=42.")

output_file = "F:/Full Stack/spendsmart-ml/notebooks/SpendSmart_Research_Complete.ipynb"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully.")
