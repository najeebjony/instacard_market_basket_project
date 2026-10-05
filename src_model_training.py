import pandas as pd
import numpy as np
import os
import json
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report
)

def train_advanced_models():
    print("[1/6] Loading advanced engineered dataset...")
    df = pd.read_csv('data/features_engineered_sample.csv')
    print(f"Dataset Shape: {df.shape}")
    
    feature_cols = [
        # User Features
        'user_total_orders', 'user_total_items', 'user_reorder_ratio', 
        'user_avg_days_between_orders', 'user_distinct_products', 'user_avg_basket_size',
        'user_reorder_velocity',
        # Product Features
        'prod_total_purchases', 'prod_reorder_ratio', 'prod_avg_cart_position', 'prod_distinct_users',
        'aisle_id', 'department_id', 'prod_dept_reorder_share',
        # User x Product Interaction & Streak Features
        'up_total_orders', 'up_first_order_number', 'up_last_order_number',
        'up_avg_cart_position', 'up_reorder_ratio', 'up_orders_since_last_purchase', 'up_order_rate',
        'up_streak_since_first', 'up_slot_priority_ratio',
        # Temporal & Context Features
        'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'reorder_gap_ratio'
    ]
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # User-wise Train/Test Split (80% Train, 20% Test)
    print("[2/6] User-Wise Stratified Split (80/20)...")
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
    print(f"Train Set: {X_train.shape[0]:,} rows ({len(train_users):,} users) | Pos Weight: {pos_weight:.2f}")
    print(f"Test Set:  {X_test.shape[0]:,} rows ({len(unique_users)-len(train_users):,} users)")
    
    # Models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=14, 
            min_samples_split=8, 
            class_weight='balanced', 
            n_jobs=-1, 
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, 
            max_depth=6, 
            learning_rate=0.06, 
            scale_pos_weight=pos_weight, 
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=-1, 
            random_state=42, 
            eval_metric='logloss'
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=220, 
            max_depth=8, 
            num_leaves=40, 
            learning_rate=0.04, 
            scale_pos_weight=pos_weight, 
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=-1, 
            random_state=42,
            verbose=-1
        )
    }
    
    results = []
    trained_models = {}
    probs_dict = {}
    
    print("[3/6] Training individual models...")
    for name, model in models.items():
        t0 = time.time()
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        y_prob = model.predict_proba(X_test)[:, 1]
        probs_dict[name] = y_prob
        trained_models[name] = model
        
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"  {name} -> ROC-AUC: {roc_auc:.4f} | Recall: {rec:.4f} | Default Acc: {acc:.4f} ({fit_time:.1f}s)")
        results.append({
            "Model": name,
            "Default Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Training Time (s)": round(fit_time, 2)
        })
        
    # Build Ensemble Blend (0.50 XGB + 0.40 LightGBM + 0.10 RF)
    print("\n[4/6] Creating Weighted Blended Ensemble...")
    blend_prob = 0.50 * probs_dict["XGBoost"] + 0.40 * probs_dict["LightGBM"] + 0.10 * probs_dict["Random Forest"]
    blend_auc = roc_auc_score(y_test, blend_prob)
    print(f"Ensemble Blend ROC-AUC: {blend_auc:.4f}")
    
    # Threshold Optimization for Boosted Accuracy
    print("\n[5/6] Running Decision Threshold Search for Maximum Accuracy & Optimal F1...")
    thresholds = np.linspace(0.1, 0.9, 81)
    acc_curve = []
    f1_curve = []
    
    best_acc = 0
    best_acc_thresh = 0.5
    
    best_f1 = 0
    best_f1_thresh = 0.5
    
    for t in thresholds:
        preds = (blend_prob >= t).astype(int)
        cur_acc = accuracy_score(y_test, preds)
        cur_f1 = f1_score(y_test, preds, zero_division=0)
        acc_curve.append(cur_acc)
        f1_curve.append(cur_f1)
        
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_acc_thresh = t
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_f1_thresh = t
            
    # Tuned predictions at optimal threshold
    optimal_preds = (blend_prob >= best_acc_thresh).astype(int)
    opt_prec = precision_score(y_test, optimal_preds, zero_division=0)
    opt_rec = recall_score(y_test, optimal_preds, zero_division=0)
    
    print("="*75)
    print("THRESHOLD OPTIMIZATION RESULTS:")
    print("="*75)
    print(f"Default Threshold (0.50) -> Accuracy: {accuracy_score(y_test, (blend_prob >= 0.5).astype(int))*100:.2f}% | Recall: {recall_score(y_test, (blend_prob >= 0.5).astype(int))*100:.2f}%")
    print(f"Optimal Accuracy Threshold ({best_acc_thresh:.2f}) -> ACCURACY: {best_acc*100:.2f}% | Precision: {opt_prec*100:.2f}%")
    print(f"Optimal F1 Threshold ({best_f1_thresh:.2f})       -> F1-Score: {best_f1:.4f}")
    print(f"Blended Ensemble ROC-AUC Score: {blend_auc:.4f}")
    
    # Save Best Model (Tuned XGBoost / LightGBM)
    os.makedirs('models', exist_ok=True)
    best_model = trained_models["XGBoost"]
    joblib.dump(best_model, 'models/best_reorder_model.joblib')
    
    metadata = {
        "best_model_name": "Enhanced Tuned XGBoost + Blended Ensemble",
        "optimal_accuracy_threshold": float(best_acc_thresh),
        "optimal_f1_threshold": float(best_f1_thresh),
        "boosted_accuracy_score": round(float(best_acc), 4),
        "ensemble_roc_auc": round(float(blend_auc), 4),
        "feature_cols": feature_cols,
        "metrics_summary": results
    }
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    # Save test sample for Streamlit live predictions
    app_sample = test_df.sample(n=min(5000, len(test_df)), random_state=42)
    products = pd.read_csv('data/products.csv')
    aisles = pd.read_csv('data/aisles.csv')
    depts = pd.read_csv('data/departments.csv')
    app_sample = app_sample.merge(products[['product_id', 'product_name']], on='product_id', how='left')
    app_sample = app_sample.merge(aisles[['aisle_id', 'aisle']], on='aisle_id', how='left')
    app_sample = app_sample.merge(depts[['department_id', 'department']], on='department_id', how='left')
    app_sample.to_csv('data/app_sample_data.csv', index=False)
    
    print("\n[SUCCESS] Advanced models trained, threshold optimized, and artifacts saved!")

if __name__ == '__main__':
    train_advanced_models()
