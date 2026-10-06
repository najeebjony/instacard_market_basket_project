# 🛒 Instacart Market Basket & Reorder Intelligence Platform
### *Ek End-to-End Enterprise Data Science, Machine Learning aur Interactive Analytics Portfolio Project*

[![Author](https://img.shields.io/badge/Author-Najeeb_Ullah-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/)
[![GitHub Profile](https://img.shields.io/badge/GitHub-najeebjony-181717?style=for-the-badge&logo=github)](https://github.com/najeebjony)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned_Model-EB6100?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📌 Executive Summary aur Business Impact

Grocery e-commerce (jaise Instacart) me **customer retention aur aadatan dobara order karna Gross Merchandise Value (GMV) ka 60% se zyada hissa** banata hai. Is project ka asal maqsad ek end-to-end machine learning system banana hai jo sahi andaza lagaye ke customer apne agle checkout me koi khaas product **dobara order karega ya nahi** (`reordered = 1` ya `0`).

Ye platform ye cheezein deta hai:
1. **Behavioral Exploratory Data Analysis (EDA)**: **3.4M+ orders** aur **50K+ products** par.
2. **Temporal Intelligence aur Peak Rush Analytics**: warehouse logistics aur delivery staffing ke liye.
3. **Leakage-Free Multi-Level Feature Store** (User, Product, User $\times$ Product aur Time Context features).
4. **Machine Learning Model Benchmarking aur GridSearchCV Hyperparameter Tuning** (Random Forest, XGBoost, LightGBM).
5. **Decision Threshold Optimization**: imbalanced target ke liye (threshold **0.72**: precision **25% → 40%**, F1 **0.38 → 0.44**).
6. **Production-Grade Streamlit Multi-Page Web Application**: live reorder simulation aur model ke asli (live computed) metrics ke sath.

---

## 🏗️ Project Architecture aur Pipeline Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                   INSTACART DATA ARCHITECTURE & FLOW                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       [ Raw Relational CSVs ]         [ Smart ~10% User Sampling ]
       - orders.csv (3.4M rows)        - Full User Journey Preserved
       - order_products__prior.csv     - Temporal Continuity Kept
       - order_products__train.csv     - Zero Data Leakage
                    │
                    ▼
     ┌─────────────────────────────┐
     │   Multi-Level Feature Store │
     ├─────────────────────────────┤
     │ • User Profile Features     │
     │ • Product Affinity Metrics  │
     │ • User × Product Recency    │
     │ • Temporal Order Context    │
     └──────────────┬──────────────┘
                    │
                    ▼
     ┌─────────────────────────────┐
     │  User-Wise 80/20 Train/Test │
     ├─────────────────────────────┤
     │ • Positive Weight: 9.26     │
     │ • 3-Fold Stratified CV      │
     │ • GridSearchCV Optimization │
     └──────────────┬──────────────┘
                    │
                    ▼
     ┌─────────────────────────────┐
     │  Production Model & Web App │
     ├─────────────────────────────┤
     │ • Tuned XGBoost (28 feats)  │
     │ • Threshold Tuned (0.72)    │
     │ • Streamlit Web Dashboard   │
     │ • Live Scenario Simulation  │
     └─────────────────────────────┘
```

---

## 📊 Aham Business aur Exploratory Findings

| Category | Asal Finding | Strategic / Business Mashwara |
| :--- | :--- | :--- |
| **Haftay ka Ordering Rhythm** | **Sunday (Day 0) aur Monday (Day 1)** kul orders ka ~35% hain. | Delivery fleet aur packing staff Sun/Mon ki subah sab se zyada rakho. |
| **Ghanton ka Traffic Peak** | **9:00 AM – 5:00 PM** sab se zyada rush ka waqt hai (Peak: 10 AM – 2 PM). | Scheduled ETL pipelines aur maintenance off-peak (12 AM – 6 AM) me chalao. |
| **Aadatan Reorder Cycles** | **Day 7, 14, 21 aur 30** par wazeh periodic spikes. | 7th din ki subah automated "Restock Your Essentials" push notification bhejo. |
| **Department Loyalty** | **Dairy/Eggs (67%)** aur **Produce (65%)** me sab se zyada repeat loyalty. | Produce ko competitive "hook" banao aur sath me zyada margin wale pantry aur personal care items cross-sell karo. |
| **Cart Placement Affinity** | **Position 1–3** par add hone wale items ka **>68% reorder rate** hai. | Mobile homepage par numayan **"Quick Reorder / Buy It Again"** widget lagao. |
| **Market Basket Co-Occurrence** | **Bananas + Organic Avocados** sab se aam cross-purchased jori hai. | Bundled discount do ("Breakfast Smoothie Combo") taake Average Order Value (AOV) barhe. |

---

## 🤖 Machine Learning Model Benchmarking aur Tuning

Imbalanced target (~9.74% positive reorder rate) se nipatne ke liye models ko `scale_pos_weight = 9.26` aur `class_weight='balanced'` ke sath **80/20 User-Wise Split** par train kiya gaya (10,496 train users, 2,624 test users, taake users ke darmiyan data leakage na ho). Model ko **28 engineered features** mile (User, Product, User × Product aur Time Context).

### 📋 Model Performance Benchmark Table
*(Neeche ke saray metrics default decision threshold 0.50 par hain, test set = 170,537 rows.)*

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.7735 | 0.2622 | 0.7298 | **0.3857** | 0.8333 | 46.56 s |
| **XGBoost** | 0.7631 | 0.2548 | 0.7440 | 0.3796 | 0.8349 | 7.42 s |
| **LightGBM** | 0.7588 | 0.2520 | **0.7498** | 0.3773 | **0.8353** | **6.90 s** |

Teeno models ka ROC-AUC lagbhag barabar hai (~0.835), yaani asal farq model se nahi, features aur decision threshold se aata hai.

### 🎯 Decision Threshold Optimization

Sirf ~9.7% reorders hone ki wajah se default 0.50 cutoff par bohot zyada galat alarm aate hain (precision ≈ 25%). Precision-recall curve par F1 maximize karne wala cutoff dhoondne se behtar balance milta hai:

| Model | Decision Threshold | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| XGBoost | 0.50 (default) | 25.5% | **74.4%** | 0.380 |
| Tuned XGBoost (GridSearchCV) | **0.72 (F1-optimized)** | **40.1%** | 49.8% | **0.4445** |

Tuned XGBoost (threshold 0.72) ka **ROC-AUC 0.8349** aur accuracy **87.9%** hai.

**Confusion Matrix (Threshold = 0.72, poora held-out test set: 170,537 rows):**

| | Predicted: No (0) | Predicted: Reorder (1) |
| :--- | :---: | :---: |
| **Actual: No (0)** | 141,582 (92.0% TN) | 12,337 (8.0% FP) |
| **Actual: Reorder (1)** | 8,344 (50.2% FN) | 8,274 (49.8% TP) |

- Jab model "reorder hoga" kehta hai, to wo lagbhag **10 me se 4 baar sahi** hota hai (default threshold par 4 me se 1 baar).
- Qeemat: recall ~74% se ~50% reh gaya, yaani aadhe asli reorders miss hote hain.
- **Business guidance:** jahan reorder miss karna mehnga ho (jaise "Buy It Again" suggestions) wahan kam threshold rakho. Jahan galat suggestion mehnga ho (jaise push notification ya discount) wahan zyada threshold rakho.
- Streamlit app me threshold slider **asli model** par live precision/recall/F1 dikhata hai.

> Sirf ~10% rows reorder hain, isliye hamesha "No" kehne par bhi ~90% accuracy mil jati hai. Is liye yahan accuracy ke bajaye **F1 aur precision-recall** asal metrics hain.

**Ensemble ka tajurba:** XGBoost + LightGBM + Random Forest ka average (ROC-AUC 0.8354, best F1 0.4450) single tuned XGBoost (F1 0.4445) se behtar nahi nikla, isliye production me single tuned XGBoost rakha gaya.

---

### ⚙️ Best Hyperparameters (GridSearchCV 3-Fold Stratified CV, 72 fits):
- `learning_rate`: **0.05**
- `max_depth`: **4**
- `n_estimators`: **150**
- `subsample`: **0.8**
- Best CV ROC-AUC: **0.8283** (100,000-row training subsample par)

### 🔍 Feature Importance:
Top predictive features ki live ranking Streamlit app ke **ML Benchmarking** page aur `02_model_training.ipynb` ke Section 5 me dekhi ja sakti hai (saved model se nikali jati hai). Sab se zyada asar user-product recency aur reorder history ke features ka hai, jaise `up_orders_since_last_purchase`, `up_reorder_ratio` aur `up_order_rate`.

---

## 📱 Interactive Streamlit Web Application

Interactive web dashboard me 5 modular pages hain:
1. **🏠 Executive Home:** Top-line KPI metrics, executive summary aur project architecture.
2. **📊 EDA & Category Matrix:** Top 20 items, department loyalty matrix aur co-purchase pairs ke interactive Plotly charts.
3. **⏰ Temporal Dynamics:** Day of Week trends, ghanton ke rush curves aur interactive heatmaps.
4. **🤖 ML Benchmarking:** Comparison table, interactive threshold explorer, ROC curve, feature importance aur confusion matrix, sab saved model se live compute hote hain.
5. **🔮 Real-Time Prediction Simulator:** Interactive sliders aur preset customer personas (Loyal Regular, Casual Shopper, Dormant Item), foran probability gauge charts aur strategic recommendations ke sath.

---

## 📁 Repository Directory Structure

```text
instacart_market_basket_project/
│
├── .streamlit/
│   └── config.toml                   # Custom UI theme & server configs
│
├── data/                             # Original & Processed Datasets
│   ├── aisles.csv
│   ├── departments.csv
│   ├── orders.csv
│   ├── order_products__prior.csv
│   ├── order_products__train.csv
│   ├── products.csv
│   ├── features_engineered_sample.csv
│   ├── app_sample_data.csv
│   └── dow_hour_counts.csv           # Day × Hour order counts (heatmap)
│
├── notebooks/                        # Clean, Annotated Jupyter Notebooks
│   ├── 01_deep_eda.ipynb             # Deep Exploratory Data Analysis
│   └── 02_model_training.ipynb       # ML Training, Benchmarking & GridSearchCV
│
├── models/                           # Serialized ML Artifacts
│   ├── best_reorder_model.joblib     # Tuned Production Model
│   ├── threshold.json                # Optimized decision threshold (0.72)
│   ├── metrics.json                  # Test-set metrics used by the app
│   ├── model_metadata.json           # Model configuration & evaluation metrics
│   ├── gridsearch_results.csv        # Cross-validation tuning logs
│   ├── feature_importance.png        # Feature gain visualization
│   └── roc_curves.png                # Combined ROC curves plot
│
├── app/                              # Streamlit Application
│   └── app.py                        # Multi-page interactive application
│
├── app.py                            # Root Streamlit entrypoint
├── requirements.txt                  # Python dependencies
└── README.md                         # Comprehensive project documentation
```

---

## 🚀 Locally Kaise Chalayein

### 1. Repository Clone karo aur Virtual Environment banao:
```bash
# Clone repository
git clone https://github.com/your-username/instacart-market-basket-analysis.git
cd instacart-market-basket-analysis

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# (For macOS / Linux: source venv/bin/activate)

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Streamlit Application chalao:
```bash
streamlit run app.py
```
*Browser me `http://localhost:8501` kholo.*

### 3. Jupyter Notebooks chalao:
```bash
jupyter notebook notebooks/
```

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. **Code GitHub par push karo:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Instacart End-to-End ML Portfolio by Najeeb Ullah"
   git branch -M main
   git remote add origin https://github.com/<your-username>/instacart-market-basket-analysis.git
   git push -u origin main
   ```
2. **Streamlit Cloud par deploy karo:**
   - [share.streamlit.io](https://share.streamlit.io/) par jao aur GitHub se login karo.
   - **"New App"** par click karo.
   - Apni repository select karo, Branch: `main`, Main file path: `app.py`.
   - **"Deploy!"** par click karo.

---

## 👨‍💻 Author aur Rabta

**Najeeb Ullah**  
*Senior Data Scientist & Machine Learning Engineer*  
- **LinkedIn:** [Connect on LinkedIn](https://linkedin.com/)  
- **GitHub:** [Follow on GitHub](https://github.com/)  
- **Portfolio:** [Data Science Portfolio](https://github.com/)  

---
*Developed with ❤️ by **Najeeb Ullah**.*
