import json
import os

def make_modeling_notebook():
    cells = []

    def md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Header
    md("""# 🤖 Instacart Reorder Prediction — Machine Learning Modeling, GridSearchCV & Benchmarking
### **Author:** Najeeb Ullah (Senior Data Scientist & ML Engineer)
### **Problem Statement:**
Predict whether a customer will reorder a specific product in their next grocery checkout (`reordered = 1` or `0`), formulated as an imbalanced binary classification task with temporal user-wise validation.

---
### **Pipeline Workflow:**
1. **Ingest Feature-Engineered Dataset**
2. **User-Wise Stratified Train/Test Split (Preventing Cross-User Leakage)**
3. **Class Imbalance Strategy (`scale_pos_weight` & `class_weight='balanced'`)**
4. **Model Training & Benchmarking (Random Forest, XGBoost, LightGBM)**
5. **Hyperparameter Optimization with GridSearchCV (3-Fold Cross Validation)**
6. **Multi-Model ROC Curves & Confusion Matrix Comparison**
7. **Feature Importance Interpretability (Top 15 Drivers)**
8. **Artifact Serialization (`.joblib` Export for Streamlit Web App)**""")

    # Cell 1: Imports
    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import time

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print("✅ Modeling packages successfully imported.")""")

    # Cell 2: Data Loading
    md("""## 1. Data Ingestion & Feature Definition
Hum step 3 me create kiya gaya feature engineered dataset load karte hain.""")

    code("""df = pd.read_csv('../data/features_engineered_sample.csv')
print(f"Dataset Shape: {df.shape}")

feature_cols = [
    'user_total_orders', 'user_total_items', 'user_reorder_ratio', 
    'user_avg_days_between_orders', 'user_distinct_products', 'user_avg_basket_size',
    'prod_total_purchases', 'prod_reorder_ratio', 'prod_avg_cart_position', 'prod_distinct_users',
    'aisle_id', 'department_id',
    'up_total_orders', 'up_first_order_number', 'up_last_order_number',
    'up_avg_cart_position', 'up_reorder_ratio', 'up_orders_since_last_purchase', 'up_order_rate',
    'order_dow', 'order_hour_of_day', 'days_since_prior_order'
]

df[feature_cols] = df[feature_cols].fillna(0)
print(f"Total Features: {len(feature_cols)}")
print(f"Target Distribution:\\n{df['reordered'].value_counts(normalize=True)}")""")

    # Cell 3: User-Wise Split
    md("""## 2. User-Wise Stratified Train / Test Split
Random rows split karne ki bajaye hum **80% Users ko Train** aur **20% Users ko Test** me rakhte hain taake model unseen users par generalize kare.""")

    code("""unique_users = df['user_id'].unique()
np.random.seed(42)
train_users = np.random.choice(unique_users, size=int(len(unique_users) * 0.8), replace=False)
train_users_set = set(train_users)

train_df = df[df['user_id'].isin(train_users_set)]
test_df = df[~df['user_id'].isin(train_users_set)]

X_train = train_df[feature_cols]
y_train = train_df['reordered']
X_test = test_df[feature_cols]
y_test = test_df['reordered']

pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
print(f"Train Set: {X_train.shape[0]:,} rows ({len(train_users):,} users) | Imbalance Ratio: {pos_weight:.2f}")
print(f"Test Set:  {X_test.shape[0]:,} rows ({len(unique_users)-len(train_users):,} users)")""")

    # Cell 4: Model Training
    md("""## 3. Training 3 Algorithms (Random Forest, XGBoost, LightGBM)
Imbalance handle karne ke liye `scale_pos_weight` aur `class_weight='balanced'` configure kiya gaya hai.""")

    code("""models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, 
        max_depth=12, 
        min_samples_split=10, 
        class_weight='balanced', 
        n_jobs=-1, 
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=150, 
        max_depth=6, 
        learning_rate=0.08, 
        scale_pos_weight=pos_weight, 
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1, 
        random_state=42, 
        eval_metric='logloss'
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200, 
        max_depth=7, 
        num_leaves=31, 
        learning_rate=0.05, 
        scale_pos_weight=pos_weight, 
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1, 
        random_state=42,
        verbose=-1
    )
}

results = []
trained_models = {}
roc_data = {}

for name, model in models.items():
    t0 = time.time()
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, roc_auc)
    trained_models[name] = model
    
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Training Time (s)": round(fit_time, 2)
    })

comparison_df = pd.DataFrame(results)
print("\\n=== Model Comparison Table ===")
display(comparison_df) if 'display' in globals() else print(comparison_df)""")

    # Cell: GridSearchCV Tuning
    md("""## 4. Hyperparameter Optimization using GridSearchCV
Model ko fine-tune karne ke liye hum 3-Fold Stratified Cross-Validation ke sath hyperparameter search chalate hain.""")

    code("""from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 150],
    'subsample': [0.8, 1.0]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
xgb_base = XGBClassifier(scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss', n_jobs=-1)

# Perform GridSearchCV on validation sample
cv_sample_idx = np.random.choice(len(X_train), size=min(100000, len(X_train)), replace=False)
grid_search = GridSearchCV(estimator=xgb_base, param_grid=param_grid, scoring='roc_auc', cv=cv, verbose=1, n_jobs=-1)

print("Starting GridSearchCV...")
grid_search.fit(X_train.iloc[cv_sample_idx], y_train.iloc[cv_sample_idx])
print(f"\\nBest CV ROC-AUC: {grid_search.best_score_:.4f}")
print(f"Best Parameters: {grid_search.best_params_}")

# Fit best model on full train dataset
best_model = XGBClassifier(**grid_search.best_params_, scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss', n_jobs=-1)
best_model.fit(X_train, y_train)
best_model_name = "Tuned XGBoost (GridSearchCV)"
print("✅ Best Tuned Model fitted on full dataset.")""")

    # Cell 5: Combined ROC Curves
    md("""## 4. Multi-Model ROC Curves Comparison""")

    code("""plt.figure(figsize=(9, 6))
colors = {'Random Forest': '#2b5c8f', 'XGBoost': '#e65c00', 'LightGBM': '#27ae60'}

for name, (fpr, tpr, auc_score) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})', color=colors.get(name, 'blue'), linewidth=2.5)

plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5000)', alpha=0.6)
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Benchmark Curves', fontsize=14, fontweight='bold', pad=12)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.show()""")

    # Cell 6: Feature Importance
    md("""## 5. Feature Importance Analysis (Model Interpretability)""")

    code("""best_model_name = "XGBoost"
best_model = trained_models[best_model_name]

imp = best_model.feature_importances_
imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': imp}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df.head(15), x='Importance', y='Feature', hue='Feature', palette='mako', legend=False)
plt.title(f'Top 15 Predictive Features ({best_model_name})', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Gain / Relative Importance', fontsize=11)
plt.ylabel('Feature Name', fontsize=11)
plt.tight_layout()
plt.show()""")

    # Cell 7: Confusion Matrix
    md("""## 6. Confusion Matrix & Threshold Tuning""")

    code("""y_prob_best = best_model.predict_proba(X_test)[:, 1]
y_pred_best = (y_prob_best >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0 (No)', 'Predicted 1 (Reorder)'],
            yticklabels=['Actual 0 (No)', 'Actual 1 (Reorder)'])
plt.title(f'Confusion Matrix — {best_model_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()""")

    # Cell 8: Save Model
    md("""## 7. Model Serialization & Export for Streamlit App""")

    code("""import os
os.makedirs('../models', exist_ok=True)
joblib.dump(best_model, '../models/best_reorder_model.joblib')
print("✅ Best Model serialized to ../models/best_reorder_model.joblib")""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    output_path = os.path.join("notebooks", "02_model_training.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)
    print(f"[SUCCESS] Notebook successfully created at: {output_path}")

if __name__ == "__main__":
    make_modeling_notebook()
