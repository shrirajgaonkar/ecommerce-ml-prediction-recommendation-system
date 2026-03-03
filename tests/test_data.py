import pytest
import pandas as pd
from datetime import datetime
from src.ingest import generate_synthetic_data

def test_synthetic_data_generation():
    df = generate_synthetic_data(num_records=100)
    
    assert len(df) == 100
    expected_cols = ['event_time', 'user_id', 'product_id', 'event_type', 
                     'price', 'category_id', 'brand', 'session_id']
    for col in expected_cols:
        assert col in df.columns
        
    # Check no negative prices
    assert all(df['price'] > 0)
    
    # Check event types are well-defined
    assert set(df['event_type'].unique()).issubset({'view', 'add_to_cart', 'purchase'})
    
def test_data_types_and_missing():
    df = generate_synthetic_data(num_records=50)
    
    assert pd.api.types.is_datetime64_any_dtype(df['event_time'])
    assert pd.api.types.is_numeric_dtype(df['price'])
    assert not df.isnull().any().any() # Ensure synthetic data has no missing fields
