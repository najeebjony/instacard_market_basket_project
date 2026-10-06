import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

# ---------------------------------------------------------
# Page Configuration & Global Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Instacart Market Basket & Reorder AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI cards and typography
st.markdown("""
<style>
    .main { background-color: #FAFAFB; }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-high {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-low {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .footer {
        text-align: center;
        padding: 30px 0 10px 0;
        color: #64748B;
        font-size: 14px;
        border-top: 1px solid #E2E8F0;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data & Model Caching
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_reorder_model.joblib')
    if not os.path.exists(model_path):
        model_path = 'models/best_reorder_model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_metadata():
    meta_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_metadata.json')
    if not os.path.exists(meta_path):
        meta_path = 'models/model_metadata.json'
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_sample_dataset():
    sample_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'app_sample_data.csv')
    if not os.path.exists(sample_path):
        sample_path = 'data/app_sample_data.csv'
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    return pd.DataFrame()

@st.cache_data
def load_catalog_data():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    if not os.path.exists(os.path.join(base_dir, 'products.csv')):
        base_dir = 'data'
    
    prods = pd.read_csv(os.path.join(base_dir, 'products.csv'))
    aisles = pd.read_csv(os.path.join(base_dir, 'aisles.csv'))
    depts = pd.read_csv(os.path.join(base_dir, 'departments.csv'))
    full_cat = prods.merge(aisles, on='aisle_id', how='left').merge(depts, on='department_id', how='left')
    return full_cat

model = load_model()
metadata = load_metadata()
sample_df = load_sample_dataset()
catalog_df = load_catalog_data()


@st.cache_data
def load_json_file(name):
    """models/ folder se koi bhi json file (metrics.json, threshold.json) load karta hai."""
    p = os.path.join(os.path.dirname(__file__), '..', 'models', name)
    if not os.path.exists(p):
        p = os.path.join('models', name)
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_heatmap_counts():
    """data/dow_hour_counts.csv: rows = day of week (0-6), columns = hour (0-23)."""
    for base in (os.path.join(os.path.dirname(__file__), '..', 'data'), 'data'):
        p = os.path.join(base, 'dow_hour_counts.csv')
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0)
    return None

@st.cache_data(show_spinner=False)
def compute_sample_scores(_model, _sample, feats_key):
    feats = list(feats_key)
    X = _sample[feats].fillna(0)
    return _sample['reordered'].values.astype(int), _model.predict_proba(X)[:, 1]

def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"threshold": float(t), "accuracy": (tp + tn) / len(y), "precision": prec,
            "recall": rec, "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}

def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    return float(th[int(np.argmax(f1[:-1]))])

# Model ki asal feature list (training wali) aur held-out test sample ka live scoring
if model is not None and hasattr(model, 'feature_names_in_'):
    MODEL_FEATURES = list(model.feature_names_in_)
else:
    MODEL_FEATURES = list(metadata.get('feature_cols', []))

SAMPLE_OK = (model is not None and not sample_df.empty and 'reordered' in sample_df.columns
             and len(MODEL_FEATURES) > 0 and all(f in sample_df.columns for f in MODEL_FEATURES))
if SAMPLE_OK:
    Y_S, P_S = compute_sample_scores(model, sample_df, tuple(MODEL_FEATURES))
else:
    Y_S, P_S = None, None

metrics_file = load_json_file('metrics.json')

THRESHOLD = load_json_file('threshold.json').get('threshold')
if THRESHOLD is None:
    THRESHOLD = best_f1_threshold(Y_S, P_S) if SAMPLE_OK else 0.5
THRESHOLD = float(THRESHOLD)

if metrics_file.get('final_tuned'):
    HEADLINE = dict(metrics_file['final_tuned'])
    HEADLINE['source'] = "full test set"
elif SAMPLE_OK:
    HEADLINE = metrics_at(Y_S, P_S, THRESHOLD)
    HEADLINE['roc_auc'] = float(roc_auc_score(Y_S, P_S))
    HEADLINE['source'] = f"{len(Y_S):,}-row test sample"
else:
    HEADLINE = None


# ---------------------------------------------------------
# Sidebar Navigation & Portfolio Profile
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081840.png", width=70)
    st.title("🛒 Instacart AI Platform")
    st.caption("End-to-End Market Basket & Predictive Analytics")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Executive Home", "📊 EDA & Category Matrix", "⏰ Temporal Dynamics", "🤖 ML Benchmarking", "🔮 Real-Time Prediction"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("👨‍💻 Project Author")
    st.write("**Najeeb Ullah**")
    st.caption("Senior Data Scientist | Machine Learning Engineer")
    st.markdown("""
    [![GitHub](https://img.shields.io/badge/GitHub-najeebjony-181717?style=flat&logo=github)](https://github.com/najeebjony)
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Najeeb_Ullah-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/)
    """)
    st.info("💡 **Tech Stack:** Python, Pandas, XGBoost, LightGBM, Plotly, Streamlit, GridSearchCV")


# =========================================================
# PAGE 1: EXECUTIVE HOME
# =========================================================
if page == "🏠 Executive Home":
    st.title("🛒 Instacart Market Basket & Reorder Intelligence")
    st.markdown("#### *An Enterprise-Grade Machine Learning & Behavioral Analytics Platform*")
    st.write("Ye platform customers ki purchase history, temporal ordering rhythms, aur cart dynamics ko analyze karke accurate prediction karta hai ke konsa customer konsi grocery dobara reorder karega.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3.4 Million+</div>
            <div class="metric-label">Orders in Source Dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">49,688</div>
            <div class="metric-label">Active Grocery Products</div>
        </div>
        """, unsafe_allow_html=True)
    auc_txt = f"{HEADLINE['roc_auc']:.4f}" if HEADLINE and HEADLINE.get('roc_auc') else "N/A"
    f1_txt = f"{HEADLINE['f1']:.3f}" if HEADLINE else "N/A"
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{auc_txt}</div>
            <div class="metric-label">ROC-AUC (Held-out Users)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{f1_txt}</div>
            <div class="metric-label">F1-Score (Tuned Threshold)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.2, 0.8])
    with col_left:
        st.subheader("🎯 Problem Statement & Business Objective")
        st.write("""
        Grocery e-commerce me user retention aur repeat purchases direct revenue driver hoti hain. Instacart par customers hazaron items browse karte hain. 
        
        **Business Goal:**
        - User ke cart open karte hi unhe unke **frequently reordered items** prioritize karke show karna.
        - Checkout friction kam karna aur Average Order Value (AOV) barhana.
        - Precision supply-chain forecasting aur automated replenishment push notifications trigger karna.
        """)
        
        st.subheader("🏗️ Architecture & Pipeline Flow")
        st.markdown("""
        1. **Smart User-Stratified Sampling:** ~13,000 users (10% of train users) ki 100% full temporal history preserve ki gayi.
        2. **Multi-Level Feature Store:** User Profile, Product Affinity, User $\\times$ Product Recency, aur Time Context.
        3. **Leakage-Free Validation:** Strict User-wise 80/20 train/test split.
        4. **Imbalance-Aware Modeling:** GridSearchCV tuned XGBoost, LightGBM, Random Forest with `scale_pos_weight`.
        """)
        
    with col_right:
        st.subheader("💡 Key Business Findings")
        st.success("**Peak Orders Window:** Sunday & Monday mornings (9 AM - 2 PM). Logistics capacity should peak here.")
        st.info("**High-Retention Anchor:** Dairy/Eggs (67%) and Produce (65%) have the highest repeat loyalty.")
        st.warning("**Cart Position Effect:** Position 1-3 items have a ~68% reorder likelihood (Core Staples).")
        st.error("**Reorder Spikes:** Strong peaks on Day 7, 14, 21, and 30 indicating strict restocking routines.")


# =========================================================
# PAGE 2: EDA & CATEGORY MATRIX
# =========================================================
elif page == "📊 EDA & Category Matrix":
    st.title("📊 Exploratory Data Analysis & Product Intelligence")
    st.write("Interactive deep dive into product popularity, department retention rates, and market basket co-occurrence.")
    
    tab1, tab2, tab3 = st.tabs(["🏆 Top Selling Products", "🏢 Department Loyalty Matrix", "🧺 Market Basket Top Pairs"])
    
    with tab1:
        st.subheader("Top 20 Most Ordered Grocery Items")
        top_prods_data = pd.DataFrame({
            "Product": ["Banana", "Bag of Organic Bananas", "Organic Strawberries", "Organic Baby Spinach", "Organic Hass Avocado",
                        "Organic Avocado", "Large Lemon", "Strawberries", "Limes", "Organic Whole Milk", "Organic Raspberries",
                        "Organic Yellow Onion", "Organic Garlic", "Organic Zucchini", "Organic Blueberries", "Cucumber Kirby",
                        "Organic Fuji Apple", "Organic Lemon", "Apple Honeycrisp Organic", "Organic Gala Apples"],
            "Units Ordered": [472565, 379450, 264683, 241921, 213584, 176815, 152657, 142951, 140627, 137905, 137057, 113426,
                              109778, 104823, 100043, 97315, 89632, 87746, 85020, 72846]
        })
        fig_top = px.bar(
            top_prods_data, x="Units Ordered", y="Product", orientation="h",
            color="Units Ordered", color_continuous_scale="Viridis",
            title="<b>Top 20 Products by Sales Volume</b>"
        )
        fig_top.update_layout(yaxis={'categoryorder':'total ascending'}, height=600, template='plotly_white')
        st.plotly_chart(fig_top, use_container_width=True)
        st.caption("💡 **Takeaway:** Fresh Produce & Organic items dominate the top 20, proving Instacart is primarily used for fresh perishables.")

    with tab2:
        st.subheader("Department Volume vs. Customer Reorder Loyalty")
        dept_data = pd.DataFrame({
            "Department": ["dairy eggs", "produce", "beverages", "bakery", "snacks", "deli", "frozen", "meat seafood", "pantry", "personal care", "household"],
            "Order Volume": [5372304, 9479291, 2690129, 1176787, 2887550, 1051249, 2236432, 708931, 1875369, 447123, 738666],
            "Reorder Rate (%)": [66.99, 64.99, 65.34, 62.81, 57.41, 60.77, 54.26, 56.76, 34.67, 32.11, 28.24]
        })
        fig_dept = px.scatter(
            dept_data, x="Order Volume", y="Reorder Rate (%)", size="Order Volume",
            color="Department", text="Department",
            title="<b>Department Strategic Positioning Matrix</b>"
        )
        fig_dept.update_traces(textposition='top center')
        fig_dept.update_layout(height=500, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_dept, use_container_width=True)
        st.caption("💡 **Takeaway:** Dairy/Eggs & Produce are high-frequency hook departments. Pantry and Household are low-reorder categories suitable for high margin cross-selling.")

    with tab3:
        st.subheader("Top Frequently Bought Together Pairs (Market Basket Affinity)")
        pairs_data = pd.DataFrame({
            "Pair": ["Banana + Organic Avocado", "Bag of Organic Bananas + Organic Strawberries",
                     "Bag of Organic Bananas + Organic Hass Avocado", "Banana + Organic Strawberries",
                     "Organic Strawberries + Organic Hass Avocado", "Banana + Large Lemon",
                     "Banana + Organic Baby Spinach", "Bag of Organic Bananas + Organic Baby Spinach",
                     "Banana + Strawberries", "Organic Avocado + Large Lemon"],
            "Co_occurrence": [25840, 24120, 22890, 21450, 18900, 17820, 16940, 16210, 15400, 14900]
        })
        fig_pairs = px.bar(
            pairs_data, x="Co_occurrence", y="Pair", orientation="h",
            color="Co_occurrence", color_continuous_scale="Sunset",
            title="<b>Top Co-Purchased Grocery Pairs</b>"
        )
        fig_pairs.update_layout(yaxis={'categoryorder':'total ascending'}, height=500, template='plotly_white')
        st.plotly_chart(fig_pairs, use_container_width=True)
        st.caption("💡 **Takeaway:** Cross-product bundles (e.g. Avocado + Banana Combo) can increase basket size and Average Order Value.")


# =========================================================
# PAGE 3: TEMPORAL DYNAMICS
# =========================================================
elif page == "⏰ Temporal Dynamics":
    st.title("⏰ Temporal Ordering Rhythms & Peak Traffic Analysis")
    st.write("Customer ordering habits broken down by day of the week, hour of the day, and habitual restocking cycles.")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Orders by Day of Week")
        dow_df = pd.DataFrame({
            "Day": ["Sunday (0)", "Monday (1)", "Tuesday (2)", "Wednesday (3)", "Thursday (4)", "Friday (5)", "Saturday (6)"],
            "Orders": [600802, 587450, 467260, 439640, 426339, 453379, 448761]
        })
        fig_dow = px.bar(dow_df, x="Day", y="Orders", color="Orders", color_continuous_scale="Teal", title="<b>Weekly Order Distribution</b>")
        fig_dow.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_dow, use_container_width=True)
        
    with col_t2:
        st.subheader("Orders by Hour of Day")
        hours = list(range(24))
        hourly_counts = [21875, 11580, 6939, 5162, 5342, 8879, 29068, 89242, 171842, 245452, 276144, 273611, 261886, 266081, 268905, 266214, 252615, 208845, 163617, 125610, 97621, 79560, 63462, 40211]
        fig_hour = px.line(x=hours, y=hourly_counts, markers=True, labels={'x': 'Hour of Day (24-Hour)', 'y': 'Total Orders'}, title="<b>Hourly Traffic Curve</b>")
        fig_hour.add_vrect(x0=9, x1=17, fillcolor="orange", opacity=0.15, line_width=0, annotation_text="Peak Window (9 AM - 5 PM)")
        fig_hour.update_layout(template="plotly_white")
        st.plotly_chart(fig_hour, use_container_width=True)
        
    st.subheader("Day of Week vs. Hour of Day Heatmap")
    hm = load_heatmap_counts()
    if hm is not None and hm.shape == (7, 24):
        fig_heat = px.imshow(
            hm.values,
            labels=dict(x="Hour of Day", y="Day of Week", color="Orders"),
            x=[f"{h}:00" for h in range(24)],
            y=["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            color_continuous_scale="YlGnBu",
            title="<b>Traffic Hotspots: Day of Week × Hour of Day</b>"
        )
        fig_heat.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Heatmap data file `data/dow_hour_counts.csv` nahi mili (7 rows × 24 columns chahiye). "
                   "Notebook me `pd.crosstab(orders['order_dow'], orders['order_hour_of_day']).to_csv('../data/dow_hour_counts.csv')` chala kar banao.")
    
    st.info("💡 **Executive Takeaway:** Sunday 9:00 AM - 2:00 PM aur Monday 9:00 AM - 12:00 PM peak rush period hain. Real-time batch delivery fleet allocation aur warehouse personnel shift planning isi schedule ke mutabiq optimize ki jani chahiye.")


# =========================================================
# PAGE 4: ML BENCHMARKING
# =========================================================
elif page == "🤖 ML Benchmarking":
    st.title("🤖 Machine Learning Model Benchmarking & Interpretability")
    st.write("Random Forest, XGBoost aur LightGBM ka muqabla, GridSearchCV tuning aur decision threshold optimization. "
             "Neeche ke numbers saved model aur held-out test users se nikalay gaye hain.")

    def fmt_row(name, m):
        return {
            "Model": name,
            "Threshold": round(float(m.get("threshold", 0.5)), 3),
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision": f"{m['precision'] * 100:.1f}%",
            "Recall": f"{m['recall'] * 100:.1f}%",
            "F1-Score": f"{m['f1']:.4f}",
            "ROC-AUC": m.get("roc_auc", "-"),
        }

    # ---- 1. Comparison table
    st.subheader("📋 Model Performance Benchmark")
    rows = []
    if metrics_file.get("default_0.5"):
        rows = [fmt_row(n, m) for n, m in metrics_file["default_0.5"].items()]
        if metrics_file.get("final_tuned"):
            rows.append(fmt_row("⚡ Tuned XGBoost @ optimized threshold", metrics_file["final_tuned"]))
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Source: `models/metrics.json` (poora held-out test set).")
    elif SAMPLE_OK:
        auc_s = float(roc_auc_score(Y_S, P_S))
        r1 = metrics_at(Y_S, P_S, 0.5); r1["roc_auc"] = round(auc_s, 4)
        r2 = metrics_at(Y_S, P_S, THRESHOLD); r2["roc_auc"] = round(auc_s, 4)
        st.dataframe(pd.DataFrame([fmt_row("Saved model @ default 0.50", r1),
                                   fmt_row("Saved model @ optimized threshold", r2)]),
                     use_container_width=True, hide_index=True)
        st.caption(f"Live computed on {len(Y_S):,} held-out test rows. Teeno models ka poora table dekhne ke liye `metrics.json` generate karo.")
    else:
        st.warning("Model ya test sample file nahi mili, isliye benchmark table nahi dikha sakte.")

    st.markdown("---")

    # ---- 2. Interactive threshold explorer (asal model se)
    st.subheader("🎛️ Interactive Decision Threshold Explorer")
    if SAMPLE_OK:
        base_rate = float(Y_S.mean())
        st.write(f"Threshold slide karo aur dekho precision, recall aur F1 kaise badalte hain. "
                 f"Dhyan rahe: sirf **{base_rate * 100:.1f}%** rows reorder hain, isliye hamesha 'No' bolne se bhi "
                 f"**{(1 - base_rate) * 100:.1f}%** accuracy mil jati hai. Accuracy yahan dhoka deti hai, F1 aur precision-recall dekho.")
        sim_t = st.slider("Decision Threshold (T)", 0.10, 0.95,
                          float(min(max(round(THRESHOLD, 2), 0.10), 0.95)), 0.01)
        cur = metrics_at(Y_S, P_S, sim_t)
        base = metrics_at(Y_S, P_S, 0.5)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{cur['accuracy'] * 100:.2f}%", f"{(cur['accuracy'] - base['accuracy']) * 100:+.1f} pts vs T=0.50")
        c2.metric("Precision", f"{cur['precision'] * 100:.1f}%", f"{(cur['precision'] - base['precision']) * 100:+.1f} pts vs T=0.50")
        c3.metric("Recall", f"{cur['recall'] * 100:.1f}%", f"{(cur['recall'] - base['recall']) * 100:+.1f} pts vs T=0.50")
        c4.metric("F1-Score", f"{cur['f1']:.4f}", f"{cur['f1'] - base['f1']:+.4f} vs T=0.50")

        ts = np.linspace(0.05, 0.95, 91)
        curve = pd.DataFrame([metrics_at(Y_S, P_S, t) for t in ts])
        fig_thr = go.Figure()
        fig_thr.add_trace(go.Scatter(x=curve["threshold"], y=curve["precision"], mode="lines", name="Precision", line=dict(color="#3B82F6", width=2.5)))
        fig_thr.add_trace(go.Scatter(x=curve["threshold"], y=curve["recall"], mode="lines", name="Recall", line=dict(color="#F59E0B", width=2.5)))
        fig_thr.add_trace(go.Scatter(x=curve["threshold"], y=curve["f1"], mode="lines", name="F1-Score", line=dict(color="#10B981", width=3.5)))
        fig_thr.add_vline(x=sim_t, line_dash="dash", line_color="gray")
        fig_thr.update_layout(title="<b>Precision / Recall / F1 vs Threshold</b>", xaxis_title="Decision Threshold",
                              yaxis_title="Score", template="plotly_white", height=380)
        st.plotly_chart(fig_thr, use_container_width=True)
        st.caption(f"Live computed from the saved model on {len(Y_S):,} held-out test rows (unseen users). "
                   f"Deployed (F1-optimized) threshold: **{THRESHOLD:.2f}**.")
    else:
        sim_t = None
        st.warning("Threshold explorer ke liye `models/best_reorder_model.joblib` aur `data/app_sample_data.csv` chahiye.")

    st.markdown("---")

    # ---- 3. ROC + Feature importance (asal)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("ROC Curve (Saved Model)")
        if SAMPLE_OK:
            fpr, tpr, _ = roc_curve(Y_S, P_S)
            auc_val = float(roc_auc_score(Y_S, P_S))
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Saved model (AUC = {auc_val:.4f})", line=dict(color="#10B981", width=3.5)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess (AUC = 0.5000)", line=dict(color="gray", dash="dash")))
            fig_roc.update_layout(title="<b>ROC Curve</b>", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)", template="plotly_white")
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption(f"Live computed on {len(Y_S):,} test rows. Random Forest / LightGBM ke curves notebook `02_model_training.ipynb` me hain.")
        else:
            st.info("ROC curve ke liye model aur test sample chahiye.")

    with col_m2:
        st.subheader("Top 10 Feature Importances")
        if model is not None and hasattr(model, "feature_importances_") and len(MODEL_FEATURES) == len(model.feature_importances_):
            imp = pd.Series(model.feature_importances_, index=MODEL_FEATURES).sort_values(ascending=False).head(10)
            feat_imp_df = pd.DataFrame({"Feature": imp.index, "Importance (%)": (imp.values * 100).round(2)})
            fig_imp = px.bar(feat_imp_df, x="Importance (%)", y="Feature", orientation="h", color="Importance (%)",
                             color_continuous_scale="Darkmint", title="<b>Key Predictive Drivers</b>")
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_white", showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance ke liye model file chahiye.")

    st.markdown("---")

    # ---- 4. Confusion matrix (asal, slider ke threshold par)
    st.subheader("🎯 Confusion Matrix & Reliability Scorecard")
    if SAMPLE_OK and sim_t is not None:
        st.write(f"Held-out test sample (**{len(Y_S):,} product pairs**) par model ki classification, threshold **{sim_t:.2f}** par:")
        col_cm1, col_cm2 = st.columns([1, 1.2])
        with col_cm1:
            tn, fp, fn, tp = cur["tn"], cur["fp"], cur["fn"], cur["tp"]
            cm_z = [[tn, fp], [fn, tp]]
            n0, n1 = max(tn + fp, 1), max(fn + tp, 1)
            cm_text = [[f"{tn:,}<br>({tn / n0 * 100:.1f}% TN)", f"{fp:,}<br>({fp / n0 * 100:.1f}% FP)"],
                       [f"{fn:,}<br>({fn / n1 * 100:.1f}% FN)", f"{tp:,}<br>({tp / n1 * 100:.1f}% TP)"]]
            fig_cm = px.imshow(
                cm_z,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Predicted: No (0)", "Predicted: Reorder (1)"],
                y=["Actual: No (0)", "Actual: Reorder (1)"],
                color_continuous_scale="Blues", text_auto=False
            )
            fig_cm.update_traces(text=cm_text, texttemplate="%{text}", textfont=dict(size=14, color="black"))
            fig_cm.update_layout(title=f"<b>Confusion Matrix (Threshold = {sim_t:.2f})</b>", height=380, template="plotly_white")
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_cm2:
            st.markdown("##### 🏆 Executive Audit (live numbers)")
            auc_a = float(roc_auc_score(Y_S, P_S))
            st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
                <ul style="margin:0; padding-left: 20px; line-height: 1.8;">
                    <li><b>ROC-AUC ({auc_a:.3f}):</b> Model ek reorder wale aur ek non-reorder wale item me se reorder wale ko {auc_a * 100:.1f}% baar zyada score deta hai.</li>
                    <li><b>Precision ({cur['precision'] * 100:.1f}%):</b> Jab model "reorder hoga" kehta hai, to lagbhag {cur['precision'] * 10:.1f} / 10 baar sahi hota hai.</li>
                    <li><b>Recall ({cur['recall'] * 100:.1f}%):</b> Asal reorders me se itne model pakad leta hai; baqi miss hote hain.</li>
                    <li><b>Accuracy ({cur['accuracy'] * 100:.1f}%):</b> Is imbalanced data me hamesha "No" kehne par bhi {(1 - base_rate) * 100:.1f}% milti hai, isliye is par bharosa na karo.</li>
                    <li><b>Trade-off:</b> Threshold barhane se precision barhti hai lekin recall girta hai. Slider se khud dekho.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Confusion matrix ke liye model aur test sample file chahiye.")


# =========================================================
# PAGE 5: REAL-TIME PREDICTION ENGINE
# =========================================================
elif page == "🔮 Real-Time Prediction":
    st.title("🔮 Real-Time Reorder Prediction Simulator")
    st.write("Live inference engine: Adjust customer behavioral signals and cart context to calculate reorder probability in real time.")
    
    st.markdown("---")
    
    preset_choice = st.selectbox(
        "⚡ Choose a Customer Scenario Preset:",
        ["Scenario 1: Loyal Regular Buying Weekly Staple (High Likelihood)",
         "Scenario 2: Casual Shopper Browsing Infrequent Item (Low Likelihood)",
         "Scenario 3: Frequent Buyer Trying a Dormant Item (Moderate Likelihood)",
         "Custom Scenario (Manual Input)"]
    )
    
    # Defaults
    if "Scenario 1" in preset_choice:
        d_user_orders = 35
        d_user_reorder_ratio = 0.72
        d_user_avg_gap = 7.5
        d_prod_reorder_ratio = 0.68
        d_up_total_orders = 18
        d_up_orders_since = 0
        d_up_order_rate = 0.51
        d_days_since_prior = 7.0
    elif "Scenario 2" in preset_choice:
        d_user_orders = 5
        d_user_reorder_ratio = 0.20
        d_user_avg_gap = 25.0
        d_prod_reorder_ratio = 0.32
        d_up_total_orders = 1
        d_up_orders_since = 4
        d_up_order_rate = 0.20
        d_days_since_prior = 28.0
    elif "Scenario 3" in preset_choice:
        d_user_orders = 22
        d_user_reorder_ratio = 0.55
        d_user_avg_gap = 12.0
        d_prod_reorder_ratio = 0.50
        d_up_total_orders = 4
        d_up_orders_since = 6
        d_up_order_rate = 0.18
        d_days_since_prior = 14.0
    else:
        d_user_orders = 15
        d_user_reorder_ratio = 0.50
        d_user_avg_gap = 10.0
        d_prod_reorder_ratio = 0.55
        d_up_total_orders = 5
        d_up_orders_since = 1
        d_up_order_rate = 0.33
        d_days_since_prior = 7.0

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 👤 Customer Profile")
            user_total_orders = st.slider("User Total Lifetime Orders", 1, 100, int(d_user_orders))
            user_reorder_ratio = st.slider("User Historical Reorder Ratio", 0.0, 1.0, float(d_user_reorder_ratio), 0.05)
            user_avg_days_between_orders = st.slider("User Avg Order Gap (Days)", 1.0, 30.0, float(d_user_avg_gap), 0.5)
            user_avg_basket_size = st.number_input("User Avg Basket Size", 1, 50, 10)
            user_total_items = user_total_orders * user_avg_basket_size
            user_distinct_products = int(user_total_items * (1 - user_reorder_ratio * 0.5))

        with col2:
            st.markdown("##### 📦 Product Affinity")
            prod_reorder_ratio = st.slider("Product Overall Reorder Ratio", 0.0, 1.0, float(d_prod_reorder_ratio), 0.05)
            prod_total_purchases = st.number_input("Product Total Sales Volume", 10, 500000, 45000)
            prod_avg_cart_position = st.slider("Product Avg Cart Position", 1.0, 20.0, 4.2, 0.1)
            prod_distinct_users = int(prod_total_purchases * (1 - prod_reorder_ratio * 0.4))
            aisle_id = 24
            department_id = 4

        with col3:
            st.markdown("##### 🤝 User × Product & Context")
            up_total_orders = st.slider("Times User Bought This Product", 1, int(user_total_orders), int(min(d_up_total_orders, user_total_orders)))
            up_orders_since_last_purchase = st.slider("Orders Since Last Purchase of This Item", 0, int(user_total_orders), int(d_up_orders_since))
            up_order_rate = st.slider("User Order Rate for Item (Streak)", 0.0, 1.0, float(d_up_order_rate), 0.05)
            up_avg_cart_position = st.slider("User's Avg Cart Slot for Item", 1.0, 15.0, 2.5, 0.5)
            up_last_order_number = max(1, user_total_orders - up_orders_since_last_purchase)
            up_first_order_number = max(1, up_last_order_number - up_total_orders + 1)
            up_reorder_ratio = up_total_orders / user_total_orders
            
            st.markdown("##### ⏰ Current Cart Context")
            order_dow = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][x], index=0)
            order_hour_of_day = st.slider("Hour of Day", 0, 23, 11)
            days_since_prior_order = st.slider("Days Since Prior Order", 0.0, 30.0, float(d_days_since_prior), 1.0)
            
        submit_btn = st.form_submit_button("⚡ Run Live Reorder Prediction", use_container_width=True)
        
    if submit_btn:
        # Calculate advanced interaction features
        user_reorder_velocity = float(user_reorder_ratio * user_total_orders)
        prod_dept_reorder_share = float(prod_reorder_ratio / 0.55)
        up_streak_since_first = float(up_total_orders / max(1, (user_total_orders - up_first_order_number + 1)))
        up_slot_priority_ratio = float(up_avg_cart_position / max(1, user_avg_basket_size))
        reorder_gap_ratio = float(days_since_prior_order / max(1.0, user_avg_days_between_orders))
        dept_avg_reorder_ratio = 0.55  # approximation (training me department ka average reorder ratio)

        input_data = pd.DataFrame([{
            'user_total_orders': user_total_orders,
            'user_total_items': user_total_items,
            'user_reorder_ratio': user_reorder_ratio,
            'user_avg_days_between_orders': user_avg_days_between_orders,
            'user_distinct_products': user_distinct_products,
            'user_avg_basket_size': user_avg_basket_size,
            'user_reorder_velocity': user_reorder_velocity,
            'prod_total_purchases': prod_total_purchases,
            'prod_reorder_ratio': prod_reorder_ratio,
            'prod_avg_cart_position': prod_avg_cart_position,
            'prod_distinct_users': prod_distinct_users,
            'aisle_id': aisle_id,
            'department_id': department_id,
            'prod_dept_reorder_share': prod_dept_reorder_share,
            'up_total_orders': up_total_orders,
            'up_first_order_number': up_first_order_number,
            'up_last_order_number': up_last_order_number,
            'up_avg_cart_position': up_avg_cart_position,
            'up_reorder_ratio': up_reorder_ratio,
            'up_orders_since_last_purchase': up_orders_since_last_purchase,
            'up_order_rate': up_order_rate,
            'up_streak_since_first': up_streak_since_first,
            'up_slot_priority_ratio': up_slot_priority_ratio,
            'order_dow': order_dow,
            'order_hour_of_day': order_hour_of_day,
            'days_since_prior_order': days_since_prior_order,
            'reorder_gap_ratio': reorder_gap_ratio,
            'dept_avg_reorder_ratio': dept_avg_reorder_ratio
        }])
        
        # Model jin features par train hua, wahi columns aur wahi order use karo
        if MODEL_FEATURES:
            missing = [c for c in MODEL_FEATURES if c not in input_data.columns]
            if missing:
                st.error(f"Model ko ye features chahiye jo form me nahi hain: {missing}")
                st.stop()
            input_data = input_data[MODEL_FEATURES]
            
        if model is not None:
            prob = float(model.predict_proba(input_data)[0, 1])
        else:
            # Fallback heuristic
            prob = min(0.95, max(0.05, (up_order_rate * 0.4 + (1.0 / (up_orders_since_last_purchase + 1)) * 0.4 + prod_reorder_ratio * 0.2)))
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🎯 Prediction Results & Strategic Recommendation")
        
        res_col1, res_col2 = st.columns([1, 1.2])
        with res_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': '%', 'font': {'size': 36}},
                title={'text': "<b>Reorder Likelihood</b>", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#10B981" if prob >= THRESHOLD else "#EF4444"},
                    'steps': [
                        {'range': [0, THRESHOLD * 60], 'color': "#FEE2E2"},
                        {'range': [THRESHOLD * 60, THRESHOLD * 100], 'color': "#FEF3C7"},
                        {'range': [THRESHOLD * 100, 100], 'color': "#D1FAE5"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': THRESHOLD * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with res_col2:
            st.write("### Decision Analysis")
            st.caption(f"Decision threshold: **{THRESHOLD:.2f}** (F1-optimized). Model probabilities class-weight ki wajah se calibrated nahi hain, ye reorder scores hain.")
            if prob >= THRESHOLD:
                st.markdown('<div class="badge-high">✅ HIGH REORDER AFFINITY</div>', unsafe_allow_html=True)
                st.success(f"Model calculates a **reorder score of {prob*100:.1f}%**, which is above the decision threshold, so the user is likely to reorder this product.")
                st.markdown("""
                **Recommended Action:**
                - Product ko user ke cart me **"Buy It Again"** ya **"Quick Reorder"** slot #1 par show karein.
                - Automated push reminder trigger karein agar user restock cycle cross kar raha ho.
                """)
            elif prob >= THRESHOLD * 0.6:
                st.markdown('<div class="badge-low" style="background-color:#FEF3C7; color:#92400E;">⚠️ MODERATE REORDER AFFINITY</div>', unsafe_allow_html=True)
                st.warning(f"Model calculates a **reorder score of {prob*100:.1f}%**, below the decision threshold but not far from it.")
                st.markdown("""
                **Recommended Action:**
                - Product ko related categories browsing ke time recommended carousel me display karein.
                """)
            else:
                st.markdown('<div class="badge-low">❌ LOW REORDER AFFINITY</div>', unsafe_allow_html=True)
                st.error(f"Model calculates a **reorder score of {prob*100:.1f}%**, well below the decision threshold.")
                st.markdown("""
                **Recommended Action:**
                - Do not clutter homepage; suggest alternative popular items or newly launched products instead.
                """)


# ---------------------------------------------------------
# Global Footer
# ---------------------------------------------------------
st.markdown("""
<div class="footer">
    <b>Instacart Market Basket & Reorder AI System</b> | Designed & Developed by <b>Najeeb Ullah</b><br>
    <a href="https://github.com/najeebjony" target="_blank">GitHub Profile</a> • 
    <a href="https://linkedin.com/" target="_blank">LinkedIn Profile</a> • 
    <a href="https://github.com/najeebjony/instacart-market-basket-analysis" target="_blank">Project Repository</a>
</div>
""", unsafe_allow_html=True)
