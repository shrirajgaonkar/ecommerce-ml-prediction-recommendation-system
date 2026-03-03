import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from src.utils import config
from src.db import execute_query, write_df_to_db
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_user_features(df_events: pd.DataFrame, snapshot_time: pd.Timestamp, horizon_days: int) -> pd.DataFrame:
    """
    Generates features for all users active prior to snapshot_time, 
    and checks if they purchased in the [snapshot_time, snapshot_time + horizon_days) window.
    """
    # 1. Split data strictly on snapshot time to avoid leakage
    history_df = df_events[df_events['event_time'] < snapshot_time]
    future_df = df_events[(df_events['event_time'] >= snapshot_time) & 
                          (df_events['event_time'] < snapshot_time + pd.Timedelta(days=horizon_days))]
    
    if history_df.empty:
        return pd.DataFrame()
        
    # Get all distinct users seen so far
    users = history_df['user_id'].unique()
    features = pd.DataFrame({'user_id': users})
    features['snapshot_date'] = snapshot_time.date()
    
    # 2. Target Label Calculation (Future Window)
    future_purchases = future_df[future_df['event_type'] == 'purchase']['user_id'].unique()
    features['label_purchased_next_7d'] = features['user_id'].isin(future_purchases).astype(int)
    
    # 3. Behavioral Features (History Window)
    # Recency
    last_event = history_df.groupby('user_id')['event_time'].max().reset_index()
    last_event['recency_days_since_last_event'] = (snapshot_time - last_event['event_time']).dt.total_seconds() / (24*3600)
    features = features.merge(last_event[['user_id', 'recency_days_since_last_event']], on='user_id', how='left')
    
    purchases_only = history_df[history_df['event_type'] == 'purchase']
    last_purchase = purchases_only.groupby('user_id')['event_time'].max().reset_index()
    last_purchase['recency_days_since_last_purchase'] = (snapshot_time - last_purchase['event_time']).dt.total_seconds() / (24*3600)
    # Fill missing purchases with 999
    features = features.merge(last_purchase[['user_id', 'recency_days_since_last_purchase']], on='user_id', how='left').fillna({'recency_days_since_last_purchase': 999.0})
    
    # Windows (e.g., last 7, 30 days) counts
    for days in [7, 30]:
        window_start = snapshot_time - pd.Timedelta(days=days)
        window_df = history_df[history_df['event_time'] >= window_start]
        
        # Total events
        events_count = window_df.groupby('user_id').size().rename(f'events_last_{days}_days')
        features = features.merge(events_count, on='user_id', how='left').fillna({f'events_last_{days}_days': 0})
        
        # Events by type
        pivot = pd.crosstab(window_df['user_id'], window_df['event_type'])
        for etype, colname in zip(['purchase', 'add_to_cart', 'view'], 
                                   [f'purchases_last_{days}_days', f'carts_last_{days}_days', f'views_last_{days}_days']):
            if etype in pivot.columns:
                features = features.merge(pivot[etype].rename(colname), on='user_id', how='left').fillna({colname: 0})
            else:
                features[colname] = 0.0
                
    # Derived Rates (using 7d as an example)
    features['conversion_rate_7d'] = features['purchases_last_7_days'] / (features['views_last_7_days'] + 1)
    features['cart_rate_7d'] = features['carts_last_7_days'] / (features['views_last_7_days'] + 1)
    
    # Monetary and Product Variety
    window_30_purchases = history_df[(history_df['event_time'] >= snapshot_time - pd.Timedelta(days=30)) & (history_df['event_type'] == 'purchase')]
    avg_price = window_30_purchases.groupby('user_id')['price'].mean().rename('avg_purchase_price_30d')
    features = features.merge(avg_price, on='user_id', how='left').fillna({'avg_purchase_price_30d': 0.0})
    
    window_30_views = history_df[(history_df['event_time'] >= snapshot_time - pd.Timedelta(days=30)) & (history_df['event_type'] == 'view')]
    unique_views = window_30_views.groupby('user_id')['product_id'].nunique().rename('unique_products_viewed_30d')
    features = features.merge(unique_views, on='user_id', how='left').fillna({'unique_products_viewed_30d': 0})
    
    return features

def run_feature_engineering():
    logger.info("Loading events from database...")
    df_events = execute_query("SELECT * FROM events_raw", as_df=True)
    df_events['event_time'] = pd.to_datetime(df_events['event_time'])
    
    min_time = df_events['event_time'].min()
    max_time = df_events['event_time'].max()
    
    # We will generate snapshots. For performance, we generate weekly snapshots starting from day 30
    start_snapshot = min_time + pd.Timedelta(days=30)
    horizon = config['features']['snapshot_days'] # 7 days
    
    all_features = []
    
    current_snap = start_snapshot
    while current_snap < max_time - pd.Timedelta(days=horizon):
        logger.info(f"Generating features for snapshot: {current_snap.date()}")
        snap_features = generate_user_features(df_events, current_snap, horizon)
        if not snap_features.empty:
            all_features.append(snap_features)
        
        current_snap += pd.Timedelta(days=7) # Weekly stride
        
    if not all_features:
        logger.warning("No features generated. Not enough data?")
        return
        
    final_df = pd.concat(all_features, ignore_index=True)
    
    logger.info(f"Generated {len(final_df)} feature rows. Saving to Database and Parquet.")
    
    # Ensure types for SQL insertion
    # Replace int64 columns with int for sqlite compatibility if needed
    final_df['snapshot_date'] = pd.to_datetime(final_df['snapshot_date'])
    
    write_df_to_db(final_df, "features_user_daily", if_exists="replace", index=False)
    
    processed_dir = Path(__file__).parent.parent / config['data']['processed_dir']
    processed_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(processed_dir / "features_user_daily.parquet", index=False)
    logger.info("Feature engineering complete.")

if __name__ == "__main__":
    run_feature_engineering()
