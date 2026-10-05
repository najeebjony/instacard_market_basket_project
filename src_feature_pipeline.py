import pandas as pd
import numpy as np
import os
import gc

def reduce_mem_usage(df):
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
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

def build_advanced_features(sample_fraction=0.10, random_state=42):
    print(f"[1/7] Ingesting orders and selecting {sample_fraction*100:.0f}% users...")
    orders = pd.read_csv('data/orders.csv')
    orders = reduce_mem_usage(orders)
    
    # Train users
    train_users = orders[orders['eval_set'] == 'train']['user_id'].unique()
    np.random.seed(random_state)
    sampled_users = np.random.choice(train_users, size=int(len(train_users) * sample_fraction), replace=False)
    sampled_users_set = set(sampled_users)
    
    orders = orders[orders['user_id'].isin(sampled_users_set)].copy()
    prior_orders = orders[orders['eval_set'] == 'prior'].copy()
    train_orders = orders[orders['eval_set'] == 'train'].copy()
    
    prior_order_ids = set(prior_orders['order_id'])
    
    print(f"[2/7] Filtering prior items for {len(sampled_users):,} sampled users...")
    prior_chunks = []
    for chunk in pd.read_csv('data/order_products__prior.csv', chunksize=1000000):
        c = chunk[chunk['order_id'].isin(prior_order_ids)]
        prior_chunks.append(c)
    priors = pd.concat(prior_chunks, ignore_index=True)
    priors = reduce_mem_usage(priors)
    del prior_chunks
    gc.collect()
    
    # Merge order information into priors
    priors = priors.merge(prior_orders[['order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']], on='order_id', how='left')
    priors = reduce_mem_usage(priors)
    
    print("[3/7] Engineering User-level features...")
    user_feats = priors.groupby('user_id').agg(
        user_total_orders=('order_number', 'max'),
        user_total_items=('product_id', 'count'),
        user_reorder_ratio=('reordered', 'mean'),
        user_avg_days_between_orders=('days_since_prior_order', 'mean'),
        user_distinct_products=('product_id', 'nunique')
    ).reset_index()
    
    # User avg basket size & velocity
    user_basket = priors.groupby(['user_id', 'order_id']).size().reset_index(name='basket_size')
    user_avg_basket = user_basket.groupby('user_id')['basket_size'].mean().reset_index(name='user_avg_basket_size')
    user_feats = user_feats.merge(user_avg_basket, on='user_id', how='left')
    user_feats['user_reorder_velocity'] = user_feats['user_reorder_ratio'] * user_feats['user_total_orders']
    user_feats = reduce_mem_usage(user_feats)
    del user_basket, user_avg_basket
    gc.collect()
    
    print("[4/7] Engineering Product & Category-level features...")
    prod_feats = priors.groupby('product_id').agg(
        prod_total_purchases=('order_id', 'count'),
        prod_reorder_ratio=('reordered', 'mean'),
        prod_avg_cart_position=('add_to_cart_order', 'mean'),
        prod_distinct_users=('user_id', 'nunique')
    ).reset_index()
    
    products = pd.read_csv('data/products.csv')
    prod_feats = prod_feats.merge(products[['product_id', 'aisle_id', 'department_id']], on='product_id', how='left')
    
    # Department average reorder rates
    dept_reorder = prod_feats.groupby('department_id')['prod_reorder_ratio'].mean().reset_index(name='dept_avg_reorder_ratio')
    prod_feats = prod_feats.merge(dept_reorder, on='department_id', how='left')
    prod_feats['prod_dept_reorder_share'] = prod_feats['prod_reorder_ratio'] / (prod_feats['dept_avg_reorder_ratio'] + 1e-5)
    prod_feats = reduce_mem_usage(prod_feats)
    
    print("[5/7] Engineering User x Product Interaction features with Momentum & Streaks...")
    up_feats = priors.groupby(['user_id', 'product_id']).agg(
        up_total_orders=('order_id', 'count'),
        up_first_order_number=('order_number', 'min'),
        up_last_order_number=('order_number', 'max'),
        up_avg_cart_position=('add_to_cart_order', 'mean'),
        up_reorder_ratio=('reordered', 'mean')
    ).reset_index()
    
    up_feats = up_feats.merge(user_feats[['user_id', 'user_total_orders', 'user_avg_basket_size', 'user_avg_days_between_orders']], on='user_id', how='left')
    up_feats['up_orders_since_last_purchase'] = up_feats['user_total_orders'] - up_feats['up_last_order_number']
    up_feats['up_order_rate'] = up_feats['up_total_orders'] / up_feats['user_total_orders']
    
    # 🌟 Advanced Feature: Order streak since first appearance
    up_feats['up_streak_since_first'] = up_feats['up_total_orders'] / (up_feats['user_total_orders'] - up_feats['up_first_order_number'] + 1)
    
    # 🌟 Advanced Feature: Slot Priority Ratio in average basket
    up_feats['up_slot_priority_ratio'] = up_feats['up_avg_cart_position'] / (up_feats['user_avg_basket_size'] + 1e-5)
    
    up_feats.drop(['user_total_orders', 'user_avg_basket_size', 'user_avg_days_between_orders'], axis=1, inplace=True)
    up_feats = reduce_mem_usage(up_feats)
    
    print("[6/7] Building Target Variable & Temporal Alignment Features...")
    data = up_feats.merge(user_feats, on='user_id', how='left')
    data = data.merge(prod_feats, on='product_id', how='left')
    data = data.merge(train_orders[['user_id', 'order_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']], on='user_id', how='left')
    
    # 🌟 Advanced Feature: Temporal Gap Alignment Ratio
    data['reorder_gap_ratio'] = data['days_since_prior_order'] / (data['user_avg_days_between_orders'] + 1e-5)
    
    # Merge train items for target labels
    train_items = pd.read_csv('data/order_products__train.csv')
    train_items_sampled = train_items[train_items['order_id'].isin(set(train_orders['order_id']))]
    train_items_sampled = train_items_sampled[['order_id', 'product_id', 'reordered']].copy()
    
    data = data.merge(train_items_sampled, on=['order_id', 'product_id'], how='left')
    data['reordered'] = data['reordered'].fillna(0).astype(np.int8)
    
    print(f"[7/7] Total Candidate Pairs: {len(data):,} | Positive Reorder Rate: {data['reordered'].mean()*100:.2f}%")
    
    output_path = 'data/features_engineered_sample.csv'
    data.to_csv(output_path, index=False)
    print(f"[SUCCESS] Advanced Feature dataset saved to: {output_path} (Shape: {data.shape})")
    return data

if __name__ == '__main__':
    build_advanced_features(sample_fraction=0.10, random_state=42)
