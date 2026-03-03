import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
from scipy.sparse import csr_matrix
from src.utils import config
from src.db import execute_query
import implicit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_recommender():
    logger.info("Loading events from database for recommender training...")
    # Load raw events
    df_events = execute_query("SELECT user_id, product_id, event_type, event_time FROM events_raw", as_df=True)
    df_events['event_time'] = pd.to_datetime(df_events['event_time'])
    
    cutoff_days = config['model_recommender']['cutoff_days']
    max_time = df_events['event_time'].max()
    cutoff_time = max_time - pd.Timedelta(days=cutoff_days)
    
    train_events = df_events[df_events['event_time'] < cutoff_time].copy()
    test_events = df_events[df_events['event_time'] >= cutoff_time].copy()
    
    # Weight interactions
    weights = config['model_recommender']['weights']
    train_events['weight'] = train_events['event_type'].map(weights).fillna(0)
    
    # Aggregate explicit/implicit score
    user_item_weights = train_events.groupby(['user_id', 'product_id'])['weight'].sum().reset_index()
    
    # Create mappings
    users = user_item_weights['user_id'].unique()
    items = user_item_weights['product_id'].unique()
    
    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for i, u in enumerate(users)}
    item_to_idx = {i: idx for idx, i in enumerate(items)}
    idx_to_item = {idx: i for idx, i in enumerate(items)}
    
    # Sparse matrix: rows are items, cols are users for `implicit` library
    row = user_item_weights['product_id'].map(item_to_idx).values
    col = user_item_weights['user_id'].map(user_to_idx).values
    data = user_item_weights['weight'].values
    
    item_user_data = csr_matrix((data, (row, col)), shape=(len(items), len(users)))
    
    # Train ALS model
    factors = config['model_recommender']['factors']
    reg = config['model_recommender']['regularization']
    iters = config['model_recommender']['iterations']
    
    # Calculate confidence alpha
    alpha = 15
    item_user_data = (item_user_data * alpha).astype('double')
    
    logger.info(f"Training Alternating Least Squares (ALS) Recommender with {factors} factors...")
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters, random_state=42)
    model.fit(item_user_data)
    
    processed_dir = Path(__file__).parent.parent / config['data']['processed_dir']
    
    # Save artifacts
    artifacts = {
        'model': model,
        'user_to_idx': user_to_idx,
        'idx_to_user': idx_to_user,
        'item_to_idx': item_to_idx,
        'idx_to_item': idx_to_item,
        'user_item_csr': item_user_data.T.tocsr() # users x items
    }
    
    artifact_path = processed_dir / "recommender_model.pkl"
    with open(artifact_path, 'wb') as f:
        pickle.dump(artifacts, f)
        
    logger.info(f"Saved recommender model to {artifact_path}.")
    
    # Save test data for evaluation
    test_events.to_parquet(processed_dir / "recommender_test_set.parquet", index=False)
    
if __name__ == "__main__":
    train_recommender()
