import pandas as pd
import numpy as np
import os
import json
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

def run_gridsearch_tuning():
    print("[1/5] Loading engineered feature dataset for GridSearchCV...")
    df = pd.read_csv('data/features_engineered_sample.csv')
    
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
    
    # User-wise split
    unique_users = df['user_id'].unique()
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
    
    # Stratified CV subsample for rapid GridSearchCV
    print("[2/5] Setting up GridSearchCV Parameter Grid...")
    # Use 100,000 samples for swift, rigorous CV grid search
    cv_sample_idx = np.random.choice(len(X_train), size=min(100000, len(X_train)), replace=False)
    X_cv = X_train.iloc[cv_sample_idx]
    y_cv = y_train.iloc[cv_sample_idx]
    
    # GridSearch on XGBoost
    xgb_base = XGBClassifier(
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 150],
        'subsample': [0.8, 1.0]
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    print(f"Starting GridSearchCV with {len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['n_estimators']) * len(param_grid['subsample'])} total candidate combinations (3-Fold CV)...")
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    t0 = time.time()
    grid_search.fit(X_cv, y_cv)
    search_time = time.time() - t0
    
    print(f"\n[SUCCESS] GridSearchCV completed in {search_time:.2f} seconds!")
    print(f"Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")
    print("Best Hyperparameters found:")
    for param, val in grid_search.best_params_.items():
        print(f"  - {param}: {val}")
        
    print("\n[3/5] Refitting Best Tuned Model on Full Training Dataset...")
    best_tuned_model = XGBClassifier(
        **grid_search.best_params_,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    best_tuned_model.fit(X_train, y_train)
    
    print("[4/5] Evaluating Tuned Model on Unseen User Test Set...")
    y_prob = best_tuned_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "="*60)
    print("TUNED XGBOOST PERFORMANCE ON TEST SET")
    print("="*60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\n[5/5] Saving Tuned Model & Artifacts...")
    joblib.dump(best_tuned_model, 'models/best_reorder_model.joblib')
    
    # Save GridSearch Results Summary
    cv_results_df = pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    cv_results_df.sort_values(by='rank_test_score', inplace=True)
    cv_results_df.to_csv('models/gridsearch_results.csv', index=False)
    
    metadata = {
        "best_model_name": "Tuned XGBoost (GridSearchCV)",
        "best_hyperparameters": grid_search.best_params_,
        "best_cv_roc_auc": float(grid_search.best_score_),
        "test_metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4)
        },
        "feature_cols": feature_cols
    }
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("[SUCCESS] Tuned model and GridSearch metadata successfully updated!")

if __name__ == '__main__':
    run_gridsearch_tuning()
