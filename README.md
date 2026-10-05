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
5. **Decision Threshold Optimization**: imbalanced target ke liye (threshold **0.88**: precision **25% → 55%**, F1 **0.38 → 0.48**).
6. **Production-Grade Streamlit Multi-Page Web Application**: live reorder probability simulation aur business decision support ke sath.

---

## 🏗️ Project Architecture aur Pipeline Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                   INSTACART DATA ARCHITECTURE & FLOW                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       [ Raw Relational CSVs ]         [ Smart 15% User Sampling ]
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
     │ • Tuned XGBoost (0.8343 AUC)│
     │ • Threshold Tuned (0.88)    │
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

Imbalanced target (~9.74% positive reorder rate) se nipatne ke liye models ko `scale_pos_weight = 9.26` aur `class_weight='balanced'` ke sath **80/20 User-Wise Split** par train kiya gaya (taake users ke darmiyan data leakage na ho).

### 📋 Model Performance Benchmark Table
*(Neeche ke saray metrics default decision threshold 0.50 par hain.)*

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.7722 | 0.2602 | 0.7261 | 0.3831 | 0.8314 | 45.96 s |
| **LightGBM** | 0.7574 | 0.2510 | **0.7505** | 0.3761 | **0.8345** | **6.87 s** |
| **XGBoost (Baseline)** | 0.7615 | 0.2538 | 0.7463 | **0.3788** | **0.8345** | 7.68 s |
| **Tuned XGBoost (GridSearchCV)** | 0.7584 | 0.2516 | 0.7494 | 0.3767 | **0.8343** | 53.23 s (72 fits) |

### 🎯 Decision Threshold Optimization

Sirf ~9.7% reorders hone ki wajah se default 0.50 cutoff par bohot zyada galat alarm aate hain (precision ≈ 25%). Precision-recall curve par probability cutoff tune karne se behtar balance milta hai:

| Decision Threshold | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: |
| 0.50 (default) | 25.4% | **74.6%** | 0.379 |
| 0.714 (max-F1 search) | 39.3% | 50.5% | 0.442 |
| **0.88 (optimized)** | **55.4%** | 42.1% | **~0.479** |

**Confusion Matrix (Threshold = 0.88):**

| | Predicted: No (0) | Predicted: Reorder (1) |
| :--- | :---: | :---: |
| **Actual: No (0)** | 148,300 (96.3% TN) | 5,627 (3.7% FP) |
| **Actual: Reorder (1)** | 9,610 (57.9% FN) | 7,000 (42.1% TP) |

- Jab model "reorder hoga" kehta hai, to wo **10 me se 5 se zyada baar sahi** hota hai (default threshold par 4 me se 1 baar).
- Galat alarm **36,455** (threshold 0.50) se ghat kar **5,627** (threshold 0.88) reh gaye.
- Qeemat: recall ~75% se 42% reh gaya, yaani kam asli reorders pakde jate hain.
- **Business guidance:** jahan reorder miss karna mehnga ho (jaise "Buy It Again" suggestions) wahan kam threshold rakho. Jahan galat suggestion mehnga ho (jaise push notification ya discount) wahan zyada threshold rakho.

> 0.88 par overall accuracy ~91% hai, lekin ~90% rows non-reorder hain, isliye yahan accuracy ke bajaye **F1 aur precision-recall** asal metrics hain.

---

### ⚙️ Best Hyperparameters (GridSearchCV 3-Fold Stratified CV):
- `learning_rate`: **0.05**
- `max_depth`: **6**
- `n_estimators`: **100**
- `subsample`: **0.8**

### 🔍 Sab se Aham Predictive Features:
1. `up_orders_since_last_purchase` (**23.03%** gain): kharidari ki recency (sab se mazboot decay signal).
2. `up_reorder_ratio` (**18.83%** gain): user ka us product se purana lagao.
3. `up_order_rate` (**17.20%** gain): customer ki basket me item ka frequency hissa.
4. `up_total_orders` (**16.29%** gain): user-product jori ke kul orders.

---

## 📱 Interactive Streamlit Web Application

Interactive web dashboard me 5 modular pages hain:
1. **🏠 Executive Home:** Top-line KPI metrics, executive summary aur project architecture.
2. **📊 EDA & Category Matrix:** Top 20 items, department loyalty matrix aur co-purchase pairs ke interactive Plotly charts.
3. **⏰ Temporal Dynamics:** Day of Week trends, ghanton ke rush curves aur interactive heatmaps.
4. **🤖 ML Benchmarking:** Multi-model ROC curves, feature importance charts aur comparison tables.
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
│   └── app_sample_data.csv
│
├── notebooks/                        # Clean, Annotated Jupyter Notebooks
│   ├── 01_deep_eda.ipynb             # Deep Exploratory Data Analysis
│   └── 02_model_training.ipynb       # ML Training, Benchmarking & GridSearchCV
│
├── models/                           # Serialized ML Artifacts
│   ├── best_reorder_model.joblib     # Tuned Production Model
│   ├── threshold.json                # Optimized decision threshold (0.88)
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
