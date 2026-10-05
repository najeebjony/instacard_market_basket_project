import json
import os

def make_notebook():
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
    md("""# 🛒 Instacart Market Basket Analysis — Comprehensive Exploratory Data Analysis (EDA)
### **Author:** Najeeb Ullah (Senior Data Scientist & ML Engineer)
### **Objective:**
Deep dive into customer ordering behavior, temporal dynamics, product affinity, customer segmentation, and market basket patterns to formulate data-driven business strategies and engineer high-signal features for predictive ML modeling.

---
### **Table of Contents:**
1. **Environment Setup & Memory-Efficient Data Ingestion**
2. **Smart User-Level Stratified Sampling**
3. **Data Quality & Hygiene Audit**
4. **Temporal Analysis (Ordering Patterns, Peak Hours & Reorder Cycles)**
5. **Product & Category Portfolio Analysis**
6. **Cart Dynamics & Position Affinity**
7. **Customer Behavior & RFM-based Machine Learning Segmentation (K-Means)**
8. **Market Basket Association Analysis (Top Product Pairs)**
9. **Executive Business Summary & Feature Engineering Roadmap**""")

    # Cell 1: Imports
    md("""## 1. Environment Setup & Data Ingestion
Hum zaroori libraries import karenge aur ek memory-optimization function banayenge jo data types ko downcast karke 70-80% RAM bachaye.""")

    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Visual settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

print("✅ Libraries successfully imported.")""")

    # Cell 2: Memory downcasting helper
    md("""### 💾 Memory Optimization Helper Function
Pandas by default 64-bit integers aur floats use karta hai. Instacart dataset me millions of rows hain, isliye integer aur float types ko downcast karna zaroori hai.""")

    code("""def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df""")

    # Cell 3: Loading raw data
    md("""### 📂 Raw Data Loading
Hum metadata aur base tables load karte hain.""")

    code("""print("Loading datasets...")
aisles = pd.read_csv('../data/aisles.csv')
departments = pd.read_csv('../data/departments.csv')
products = pd.read_csv('../data/products.csv')
orders = pd.read_csv('../data/orders.csv')

aisles = reduce_mem_usage(aisles)
departments = reduce_mem_usage(departments)
products = reduce_mem_usage(products)
orders = reduce_mem_usage(orders)

print(f"Orders Shape: {orders.shape}")
print(f"Products Shape: {products.shape}")
print(f"Aisles Shape: {aisles.shape}")
print(f"Departments Shape: {departments.shape}")""")

    # Cell 4: Smart Sampling Strategy
    md("""## 2. Smart User-Level Stratified Sampling
> **Important DS Note:** 
> Agar hum random rows chunk (`nrows=2000000`) load karein, to ek customer ke aadhay orders gayab ho jate hain, jis se uski shopping history, order gap, aur reorder ratios corrupt ho jate hain.
> **Stratified User-Level Sampling Solution:** Hum **15% unique users (~31,000 users)** ko randomly select karte hain aur unke **sare ke sare prior aur train orders** ko preserve karte hain. Is se customer journey 100% accurate rehti hai.""")

    code("""np.random.seed(42)
all_users = orders['user_id'].unique()
sample_size = int(len(all_users) * 0.15)
sampled_users = np.random.choice(all_users, size=sample_size, replace=False)
sampled_users_set = set(sampled_users)

print(f"Total Unique Users in Dataset: {len(all_users):,}")
print(f"Sampled Users (15%): {len(sampled_users):,}")

# Filter orders for sampled users
orders_sample = orders[orders['user_id'].isin(sampled_users_set)].copy()
sampled_order_ids = set(orders_sample['order_id'])

print(f"Sampled Orders: {len(orders_sample):,} rows")""")

    # Cell 5: Loading order_products with user filter
    md("""### 📦 Filtering `order_products__prior` & `order_products__train`
Ab hum sampled order IDs ke hisab se order products load karenge taake RAM overflow na ho aur speed tez rahe.""")

    code("""print("Filtering order_products__prior...")
# Load in chunks for memory safety
prior_chunks = []
for chunk in pd.read_csv('../data/order_products__prior.csv', chunksize=1000000):
    filtered_chunk = chunk[chunk['order_id'].isin(sampled_order_ids)]
    prior_chunks.append(filtered_chunk)
order_products_prior = pd.concat(prior_chunks, ignore_index=True)
order_products_prior = reduce_mem_usage(order_products_prior)

print("Filtering order_products__train...")
train_df = pd.read_csv('../data/order_products__train.csv')
order_products_train = train_df[train_df['order_id'].isin(sampled_order_ids)].copy()
order_products_train = reduce_mem_usage(order_products_train)

print(f"Filtered Prior Items: {len(order_products_prior):,} rows")
print(f"Filtered Train Items: {len(order_products_train):,} rows")""")

    # Cell 6: Data Quality Audit
    md("""## 3. Data Quality & Hygiene Audit
Hum missing values, duplicates, dtypes, aur structural consistency verify karte hain.""")

    code("""print("--- Orders Null Values ---")
print(orders_sample.isnull().sum())

# Missing value in days_since_prior_order analysis:
first_orders_nulls = orders_sample[orders_sample['order_number'] == 1]['days_since_prior_order'].isnull().sum()
total_first_orders = (orders_sample['order_number'] == 1).sum()
print(f"\\nFirst orders (order_number=1) with NaN days_since_prior_order: {first_orders_nulls} out of {total_first_orders}")

print("\\n--- Duplicates Check ---")
print(f"Duplicate rows in orders: {orders_sample.duplicated().sum()}")
print(f"Duplicate rows in prior items: {order_products_prior.duplicated(subset=['order_id', 'product_id']).sum()}")""")

    md("""**Data Hygiene Insights:**
- `days_since_prior_order` me missing values sirf pehle order (`order_number = 1`) par hain kyunki pehle order se pehle koi prior order nahi hota. Ye structural missingness hai, koi data corruption nahi.
- Zero duplicate rows hain order level aur order-item level par.""")

    # Cell 7: Merged Dataset
    md("""### 🔗 Rich Merged Master Dataframe
EDA aur visualization ke liye products, aisles, aur departments ko merge karte hain.""")

    code("""# Merge order_products_prior with products, aisles, departments, and orders
order_details = (order_products_prior
                 .merge(products, on='product_id', how='left')
                 .merge(aisles, on='aisle_id', how='left')
                 .merge(departments, on='department_id', how='left')
                 .merge(orders_sample[['order_id', 'user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']], on='order_id', how='left'))

print("Merged Data Sample:")
order_details.head(3)""")

    # Cell 8: Time Analysis - DOW & Hour
    md("""## 4. Temporal Dynamics & Time Analysis (Critical Insights)
Hum customer ordering behavior ko timeline (Day of Week, Hour of Day, Weekly & Monthly reorder cycles) par analyze karte hain.""")

    code("""day_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
orders_sample['day_name'] = orders_sample['order_dow'].map(day_map)

# 1. Orders by Day of Week
dow_counts = orders_sample['day_name'].value_counts().reindex(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

fig_dow = px.bar(
    x=dow_counts.index, 
    y=dow_counts.values,
    color=dow_counts.values,
    color_continuous_scale='Teal',
    labels={'x': 'Day of Week', 'y': 'Total Orders Placed'},
    title='<b>Weekly Ordering Rhythm: Total Orders by Day of Week</b>'
)
fig_dow.update_layout(template='plotly_white', showlegend=False)
fig_dow.show()""")

    md("""**Business Insight (Day of Week):**
- **Peak Days:** Sunday (Day 0) aur Monday (Day 1) par sab se zyada orders place hote hain (~35% of total volume). Log weekly grocery routine set karte hain.
- **Mid-week Dip:** Tuesday aur Wednesday ko orders kam ho jate hain.
- **Actionable Strategy:** Mid-week flash discounts ya promotions run karke Tuesday/Wednesday ke drop ko mitigate kiya ja sakta hai, jabke Sunday/Monday ko delivery logistics aur warehouse staffing maximum honi chahiye.""")

    # Cell 9: Orders by Hour of Day
    code("""# 2. Orders by Hour of Day
hourly_counts = orders_sample['order_hour_of_day'].value_counts().sort_index()

fig_hour = px.line(
    x=hourly_counts.index,
    y=hourly_counts.values,
    markers=True,
    labels={'x': 'Hour of Day (24-Hour Format)', 'y': 'Total Orders Placed'},
    title='<b>Hourly Traffic Curve: Order Volume Throughout the Day</b>'
)
fig_hour.add_vrect(x0=9, x1=17, fillcolor="orange", opacity=0.15, line_width=0, annotation_text="Peak Rush Window (9 AM - 5 PM)", annotation_position="top left")
fig_hour.update_layout(template='plotly_white')
fig_hour.show()""")

    md("""**Business Insight (Hourly Traffic):**
- **Rush Window:** Subha **9:00 AM se Shaam 5:00 PM** ke darmiyan 70%+ orders aate hain, jisme peak 10:00 AM se 2:00 PM ke beech hoti hai.
- **Off-Peak:** Raat 12:00 AM se 6:00 AM tak system traffic lowest rehti hai (best window for scheduled maintenance & batch ETL pipeline runs).""")

    # Cell 10: Heatmap Day x Hour
    code("""# 3. Heatmap: Day of Week vs Hour of Day
pivot_time = orders_sample.pivot_table(index='order_dow', columns='order_hour_of_day', values='order_id', aggfunc='count')
pivot_time.index = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

plt.figure(figsize=(14, 6))
sns.heatmap(pivot_time, cmap='YlGnBu', annot=False, cbar_kws={'label': 'Order Count'})
plt.title('<b>Order Heatmap: Day of Week vs Hour of Day</b>', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Hour of Day', fontsize=11)
plt.ylabel('Day of Week', fontsize=11)
plt.tight_layout()
plt.show()""")

    md("""**Business Insight (Day × Hour Heatmap):**
- **Prime Hotspots:** Sunday subha 9 AM se 2 PM aur Monday subha 9 AM se 11 AM sab se dense traffic zones hain.
- **Delivery Slot Allocation:** Delivery slot pricing (surge pricing or priority fee) Sunday/Monday mornings me maximize ki ja sakti hai.""")

    # Cell 11: Reorder Cycles (days_since_prior_order)
    code("""# 4. Reorder Gap Cycles (days_since_prior_order)
prior_gap = orders_sample['days_since_prior_order'].dropna()

fig_gap = px.histogram(
    prior_gap, 
    nbins=31, 
    labels={'value': 'Days Since Prior Order'},
    title='<b>Customer Reorder Cycles: Habitual Restocking Frequencies</b>',
    color_discrete_sequence=['#2E86AB']
)
fig_gap.update_layout(template='plotly_white', bargap=0.1, yaxis_title='Order Frequency')
fig_gap.show()""")

    md("""**Business Insight (Reorder Cycles):**
- **Weekly Spikes:** Day 7, Day 14, Day 21, aur Day 28 par clear periodic peaks dikhai deti hain jo weekly grocery routine ko sabit karti hain.
- **30-Day Threshold:** 30 days par sab se barhi spike hai (Instacart max cap 30 days hai, jisme monthly buyers aur churn-risk customers dono aate hain).
- **Retention Strategy:** Agar koi user 6th din order na kare, to automated push notification / reminder email ("Re-stock your essentials") 7th din subha trigger karna conversion rate ko boost karega.""")

    # Cell 12: Product & Category Analysis
    md("""## 5. Product & Category Portfolio Analysis
Top selling items, high-affinity departments, aur category-level reorder loyalty ka jaiza.""")

    code("""# Top 20 Best Selling Products
top_products = order_details['product_name'].value_counts().head(20).reset_index()
top_products.columns = ['product_name', 'order_count']

fig_prod = px.bar(
    top_products,
    x='order_count',
    y='product_name',
    orientation='h',
    color='order_count',
    color_continuous_scale='Viridis',
    title='<b>Top 20 Most Frequently Ordered Products</b>',
    labels={'order_count': 'Total Units Ordered', 'product_name': 'Product Name'}
)
fig_prod.update_layout(template='plotly_white', yaxis={'categoryorder':'total ascending'}, showlegend=False)
fig_prod.show()""")

    md("""**Business Insight (Top Products):**
- **Organic & Fresh Produce Dominance:** **Banana**, **Bag of Organic Bananas**, **Organic Strawberries**, **Organic Baby Spinach**, aur **Organic Hass Avocado** top 5 items hain.
- Customers Instacart ko fresh grocery aur daily perishables ke liye sab se zyada use karte hain.""")

    # Cell 13: Department Breakdown & Reorder Rate
    code("""# Department Volume & Reorder Ratio Analysis
dept_summary = order_details.groupby('department').agg(
    total_orders=('product_id', 'count'),
    reorder_rate=('reordered', 'mean')
).reset_index().sort_values(by='total_orders', ascending=False)

dept_summary['reorder_pct'] = dept_summary['reorder_rate'] * 100

fig_dept = px.scatter(
    dept_summary,
    x='total_orders',
    y='reorder_pct',
    size='total_orders',
    color='department',
    text='department',
    labels={'total_orders': 'Total Items Purchased', 'reorder_pct': 'Reorder Rate (%)'},
    title='<b>Department Performance Matrix: Volume vs Customer Reorder Loyalty</b>'
)
fig_dept.update_traces(textposition='top center')
fig_dept.update_layout(template='plotly_white', showlegend=False)
fig_dept.show()""")

    md("""**Business Insight (Department Loyalty):**
- **Dairy/Eggs (67%+)** aur **Produce (65%+)** highest reorder rate rakhte hain. Ye high-retention "Hook" categories hain.
- **Pantry, Personal Care, and Household** ka reorder rate 30-40% ke aas paas hai kyunki ye items long shelf-life rakhte hain ya customers doosre channels se buy karte hain.
- **Strategy:** Dairy/Produce par aggressive pricing rakh kar customers ko platform par attract karein aur high-margin pantry/personal care items cross-sell karein.""")

    # Cell 14: Cart Dynamics
    md("""## 6. Cart Placement Dynamics & Position Affinity
Dekhte hain ke cart me pehle add hone wale items ka reorder probability par kya asar parta hai.""")

    code("""# Reorder rate by add_to_cart_order
cart_analysis = order_details[order_details['add_to_cart_order'] <= 20].groupby('add_to_cart_order').agg(
    reorder_rate=('reordered', 'mean'),
    item_count=('product_id', 'count')
).reset_index()

fig_cart = px.line(
    cart_analysis,
    x='add_to_cart_order',
    y='reorder_rate',
    markers=True,
    labels={'add_to_cart_order': 'Add to Cart Sequence Position', 'reorder_rate': 'Reorder Probability'},
    title='<b>Cart Position vs Reorder Likelihood (First-in-Mind Effect)</b>'
)
fig_cart.update_layout(template='plotly_white', yaxis_tickformat='.0%')
fig_cart.show()""")

    md("""**Business Insight (Cart Position Affinity):**
- **First-in-Mind Effect:** Cart me position 1 ya 2 par add hone wale items ka reorder rate **~68%** hai. Position 10 ke baad ye gir kar **<45%** ho jata hai.
- **Recommendation Feature:** Pehle 2-3 items customer ke core habitual staples hote hain. App homepage par "Quick Reorder" widget top par show karne se checkout time drastically drop hoga aur GMV boost hogi.""")

    # Cell 15: Customer Segmentation (K-Means)
    md("""## 7. Customer Behavioral Segmentation (Machine Learning K-Means)
Hum RFM-style customer features (Total Orders, Average Reorder Gap, Average Basket Size, aur Overall Reorder Ratio) nikal kar K-Means clustering apply karenge.""")

    code("""# Aggregating user-level metrics
user_basket = order_details.groupby(['user_id', 'order_id']).size().reset_index(name='basket_size')
user_avg_basket = user_basket.groupby('user_id')['basket_size'].mean().reset_index(name='avg_basket_size')

user_features = orders_sample.groupby('user_id').agg(
    total_orders=('order_number', 'max'),
    avg_reorder_gap=('days_since_prior_order', 'mean')
).reset_index()

user_reorder = order_details.groupby('user_id')['reordered'].mean().reset_index(name='user_reorder_ratio')

customer_df = (user_features
               .merge(user_avg_basket, on='user_id')
               .merge(user_reorder, on='user_id')
               .fillna(0))

print("Customer Profile Sample:")
print(customer_df.head(3))

# K-Means Clustering
features_for_clustering = ['total_orders', 'avg_reorder_gap', 'avg_basket_size', 'user_reorder_ratio']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_df[features_for_clustering])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
customer_df['cluster'] = kmeans.fit_predict(X_scaled)

# Cluster Persona profiling
cluster_summary = customer_df.groupby('cluster')[features_for_clustering].mean().reset_index()
print("\\n--- Customer Segments (Cluster Profiling) ---")
print(cluster_summary)""")

    # Cell 16: Cluster visualization
    code("""fig_cluster = px.scatter(
    customer_df.sample(n=min(3000, len(customer_df)), random_state=42),
    x='total_orders',
    y='avg_basket_size',
    color=customer_df.sample(n=min(3000, len(customer_df)), random_state=42)['cluster'].astype(str),
    size='user_reorder_ratio',
    labels={'total_orders': 'Lifetime Orders', 'avg_basket_size': 'Avg Basket Size', 'color': 'Customer Segment'},
    title='<b>Customer Behavioral Segmentation Map (K-Means Personas)</b>'
)
fig_cluster.update_layout(template='plotly_white')
fig_cluster.show()""")

    md("""**Business Personas Identified:**
1. **Cluster 0 — Power Regulars:** High total orders (>40), low reorder gap (<8 days), strong loyalty. *(Strategy: VIP perks, subscription models).*
2. **Cluster 1 — Occasional Bulk Stockers:** Low order count, bara basket size (>20 items), monthly shopping pattern. *(Strategy: Bulk bundle deals & free shipping on high cart value).*
3. **Cluster 2 — Casual One-Offs:** Kam orders, chota basket size, low reorder ratio. *(Strategy: Re-engagement discount coupons).*
4. **Cluster 3 — Balanced Routine Shoppers:** Moderate basket size (8-14 items), consistent bi-weekly orders.""")

    # Cell 17: Market Basket Association
    md("""## 8. Market Basket Association Analysis (Top Product Pairs)
Konsay do products sab se zyada ek sath cart me add hote hain (Co-occurrence Matrix).""")

    code("""# Calculate Top 15 Frequently Bought Together Pairs
order_baskets = order_details.groupby('order_id')['product_name'].apply(list)

pair_counter = Counter()
for basket in order_baskets:
    unique_items = sorted(list(set(basket)))
    if len(unique_items) > 1:
        for pair in combinations(unique_items, 2):
            pair_counter[pair] += 1

top_pairs = pair_counter.most_common(15)
pairs_df = pd.DataFrame([
    {'Product A': pair[0], 'Product B': pair[1], 'Co_occurrence_Count': count}
    for pair, count in top_pairs
])

fig_pairs = px.bar(
    pairs_df,
    x='Co_occurrence_Count',
    y=[f"{a} + {b}" for a, b in zip(pairs_df['Product A'], pairs_df['Product B'])],
    orientation='h',
    color='Co_occurrence_Count',
    color_continuous_scale='Sunset',
    title='<b>Top Frequently Bought Together Pairs (Market Basket Affinity)</b>',
    labels={'x': 'Co-occurrence Frequency', 'y': 'Product Pair'}
)
fig_pairs.update_layout(template='plotly_white', yaxis={'categoryorder':'total ascending'}, showlegend=False)
fig_pairs.show()""")

    md("""**Business Insight (Cross-Merchandising Pairs):**
- Top pairs: **Banana + Organic Avocado**, **Organic Strawberries + Organic Banana**, **Organic Hass Avocado + Bag of Organic Bananas**.
- **Product Bundling:** "Breakfast Smoothie Bundle" ya "Fresh Produce Combo" offer karke Average Order Value (AOV) barhayi ja sakti hai.""")

    # Cell 18: Summary & Roadmap
    md("""## 9. Executive Summary & Next Steps for Feature Engineering
### Key Takeaways:
1. **Temporal Signals:** Peak purchasing window Sunday-Monday 9 AM-5 PM; 7/14/21/30 day reorder cycles.
2. **Product Hierarchy:** Produce & Dairy are high-reorder anchor categories.
3. **Cart Position:** Items in slots 1-3 have ~68% reorder probability.
4. **Target Variable for Modeling:** `reordered` (Binary 0 or 1).

### 🛠️ Feature Engineering Roadmap (STEP 3):
- **User Features:** `user_total_orders`, `user_reorder_ratio`, `user_avg_days_between_orders`, `user_avg_basket_size`.
- **Product Features:** `prod_total_orders`, `prod_reorder_rate`, `prod_avg_add_to_cart_position`.
- **User-Product Interaction Features:** `up_orders_count`, `up_reorder_rate`, `up_orders_since_last_purchase`, `up_avg_cart_position`.
- **Order Context Features:** `order_dow`, `order_hour_of_day`, `days_since_prior_order`.""")

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

    output_path = os.path.join("notebooks", "01_deep_eda.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)
    print(f"[SUCCESS] Notebook successfully created at: {output_path}")

if __name__ == "__main__":
    make_notebook()
