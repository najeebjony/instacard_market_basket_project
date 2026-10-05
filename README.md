# 🛒 Instacart Market Basket & Reorder Intelligence Platform
### *An End-to-End Enterprise Data Science, Machine Learning & Interactive Analytics Portfolio Project*

[![Author](https://img.shields.io/badge/Author-Najeeb_Ullah-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/)
[![GitHub Profile](https://img.shields.io/badge/GitHub-najeebjony-181717?style=for-the-badge&logo=github)](https://github.com/najeebjony)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned_Model-EB6100?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📌 Executive Summary & Business Impact

In grocery e-commerce (such as Instacart), **customer retention and habitual reorders represent over 60% of Gross Merchandise Value (GMV)**. The primary objective of this project is to build an end-to-end machine learning system that accurately predicts whether a customer will reorder a specific product in their next checkout session (`reordered = 1` or `0`).

This platform provides:
1. **Behavioral Exploratory Data Analysis (EDA)** across **3.4M+ orders** and **50K+ products**.
2. **Temporal Intelligence & Peak Rush Analytics** for warehouse logistics and delivery staffing.
3. **Leakage-Free Multi-Level Feature Store** (User, Product, User $\times$ Product, and Time Context features).
4. **Machine Learning Model Benchmarking & GridSearchCV Hyperparameter Tuning** (Random Forest, XGBoost, LightGBM).
5. **A Production-Grade Streamlit Multi-Page Web Application** featuring live reorder probability simulation and business decision support.

---

## 🏗️ Project Architecture & Pipeline Flow

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
     │ • Streamlit Web Dashboard   │
     │ • Live Scenario Simulation  │
     └─────────────────────────────┘
```

---

## 📊 Key Business & Exploratory Findings

| Category | Core Finding | Strategic / Business Recommendation |
| :--- | :--- | :--- |
| **Weekly Ordering Rhythm** | **Sunday (Day 0) & Monday (Day 1)** account for ~35% of total orders. | Delivery fleet and packing staff should be maximized on Sun/Mon mornings. |
| **Hourly Traffic Peak** | **9:00 AM – 5:00 PM** is the prime rush window (Peak: 10 AM – 2 PM). | Scheduled ETL pipelines and maintenance should run during off-peak (12 AM – 6 AM). |
| **Habitual Reorder Cycles** | Pronounced periodic spikes at **Day 7, 14, 21, and 30**. | Trigger automated "Restock Your Essentials" push notifications on the 7th day morning. |
| **Department Loyalty** | **Dairy/Eggs (67%)** & **Produce (65%)** have the highest repeat loyalty. | Use produce as a competitive "hook" while cross-selling high-margin pantry & personal care items. |
| **Cart Placement Affinity** | Items added in **positions 1–3** have a **>68% reorder rate**. | Feature a prominent **"Quick Reorder / Buy It Again"** widget on the mobile homepage. |
| **Market Basket Co-Occurrence** | **Bananas + Organic Avocados** is the most frequent cross-purchased pair. | Offer bundled discounts ("Breakfast Smoothie Combo") to raise Average Order Value (AOV). |

---

## 🤖 Machine Learning Model Benchmarking & Tuning

To tackle the **imbalanced target distribution** (~9.74% positive reorder rate), models were trained using `scale_pos_weight = 9.26` and `class_weight='balanced'` on an **80/20 User-Wise Split** (ensuring no cross-user data leakage).

### 📋 Model Performance Benchmark Table

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.7722 | 0.2602 | 0.7261 | 0.3831 | 0.8314 | 45.96 s |
| **LightGBM** | 0.7574 | 0.2510 | **0.7505** | 0.3761 | **0.8345** | **6.87 s** |
| **XGBoost (Baseline)** | 0.7615 | 0.2538 | 0.7463 | **0.3788** | **0.8345** | 7.68 s |
| **Tuned XGBoost (GridSearchCV)** | 0.7584 | 0.2516 | 0.7494 | 0.3767 | **0.8343** | 53.23 s (72 fits) |

### ⚙️ Best Hyperparameters (GridSearchCV 3-Fold Stratified CV):
- `learning_rate`: **0.05**
- `max_depth`: **6**
- `n_estimators`: **100**
- `subsample`: **0.8**

### 🔍 Top Predictive Feature Drivers:
1. `up_orders_since_last_purchase` (**23.03%** gain) — Purchase recency (strongest decay signal).
2. `up_reorder_ratio` (**18.83%** gain) — User's historic affinity for the product.
3. `up_order_rate` (**17.20%** gain) — Frequency share of the item in customer's basket.
4. `up_total_orders` (**16.29%** gain) — Lifetime order volume for the pair.

---

## 📱 Interactive Streamlit Web Application

The interactive web dashboard includes 5 modular pages:
1. **🏠 Executive Home:** Top-line KPI metrics, executive summary, and project architecture.
2. **📊 EDA & Category Matrix:** Interactive Plotly charts for top 20 items, department loyalty matrix, and co-purchase pairs.
3. **⏰ Temporal Dynamics:** Day of Week trends, hourly rush curves, and interactive heatmaps.
4. **🤖 ML Benchmarking:** Interactive multi-model ROC curves, feature importance charts, and comparison tables.
5. **🔮 Real-Time Prediction Simulator:** Interactive sliders and preset customer personas (Loyal Regular, Casual Shopper, Dormant Item) with instant probability gauge charts and strategic recommendations.

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

## 🚀 How to Run Locally

### 1. Clone the Repository & Set Up Virtual Environment:
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

### 2. Run the Streamlit Application:
```bash
streamlit run app.py
```
*Open `http://localhost:8501` in your web browser.*

### 3. Run the Jupyter Notebooks:
```bash
jupyter notebook notebooks/
```

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. **Push Code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Instacart End-to-End ML Portfolio by Najeeb Ullah"
   git branch -M main
   git remote add origin https://github.com/<your-username>/instacart-market-basket-analysis.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
   - Click **"New App"**.
   - Select your repository, Branch: `main`, Main file path: `app.py`.
   - Click **"Deploy!"**.

---

## 👨‍💻 Author & Contact

**Najeeb Ullah**  
*Senior Data Scientist & Machine Learning Engineer*  
- **LinkedIn:** [Connect on LinkedIn](https://linkedin.com/)  
- **GitHub:** [Follow on GitHub](https://github.com/)  
- **Portfolio:** [Data Science Portfolio](https://github.com/)  

---
*Developed with ❤️ by **Najeeb Ullah**.*
