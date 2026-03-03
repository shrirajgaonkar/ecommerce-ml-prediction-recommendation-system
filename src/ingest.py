import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
import logging
from src.utils import config
from src.db import init_db, write_df_to_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(num_records: int = 50000, days: int = 60) -> pd.DataFrame:
    """Generates synthetic e-commerce event data."""
    logger.info(f"Generating {num_records} synthetic rows over {days} days...")
    
    np.random.seed(42)
    random.seed(42)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate entities
    users = [f"user_{i:04d}" for i in range(2000)]
    products = [f"prod_{i:04d}" for i in range(500)]
    categories = [f"cat_{i:02d}" for i in range(20)]
    brands = [f"brand_{i:02d}" for i in range(30)]
    
    event_types = ['view', 'add_to_cart', 'purchase']
    # Probabilities for typical funnel: lots of views, fewer carts, few purchases
    event_probs = [0.80, 0.15, 0.05]
    
    prod_meta = {p: {'category_id': random.choice(categories), 'brand': random.choice(brands), 'price': round(random.uniform(10, 500), 2)} for p in products}
    
    events = []
    
    for _ in range(num_records):
        # Time distribution (random uniform within the window)
        random_seconds = random.randint(0, int((end_date - start_date).total_seconds()))
        event_time = start_date + timedelta(seconds=random_seconds)
        
        user = np.random.choice(users)
        # Power law distribution for products (some very popular)
        prod = np.random.choice(products, p=np.random.dirichlet(np.ones(len(products)), size=1)[0])
        event_type = np.random.choice(event_types, p=event_probs)
        
        meta = prod_meta[prod]
        
        events.append({
            'event_time': event_time,
            'user_id': user,
            'product_id': prod,
            'event_type': event_type,
            'price': meta['price'],
            'category_id': meta['category_id'],
            'brand': meta['brand'],
            'session_id': f"sess_{random.randint(1000, 9999)}"
        })
        
    df = pd.DataFrame(events)
    df = df.sort_values('event_time').reset_index(drop=True)
    return df

def run_ingestion():
    raw_path = Path(__file__).parent.parent / config['data']['raw_data_path']
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    if raw_path.exists():
        logger.info(f"Loading existing raw data from {raw_path}")
        df = pd.read_csv(raw_path, parse_dates=['event_time'])
    else:
        logger.info("Raw data not found. Generating synthetic data.")
        df = generate_synthetic_data(num_records=55000)
        df.to_csv(raw_path, index=False)
        logger.info(f"Saved synthetic data to {raw_path}")
        
    logger.info("Initializing database schema...")
    init_db()
    
    logger.info("Writing raw data to database...")
    write_df_to_db(df, 'events_raw', if_exists='replace', index=False)
    
    # Also populate dimension tables (basic)
    unique_users = pd.DataFrame({'user_id': df['user_id'].unique()})
    unique_users['first_seen'] = df.groupby('user_id')['event_time'].min().values
    write_df_to_db(unique_users, 'users_dim', if_exists='replace', index=False)
    
    unique_products = df.drop_duplicates('product_id')[['product_id', 'category_id', 'brand', 'price']].rename(columns={'price': 'avg_price'})
    write_df_to_db(unique_products, 'products_dim', if_exists='replace', index=False)
    
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    run_ingestion()
