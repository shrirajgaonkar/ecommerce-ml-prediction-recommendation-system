import pytest
import pandas as pd
from datetime import timedelta
from src.features import generate_user_features

@pytest.fixture
def mock_events():
    # Deterministic events to test leakage and recency
    now = pd.to_datetime('2025-01-10 12:00:00')
    data = [
        # User 1: 1 purchase at T-5, 1 view at T-1
        {'event_time': now - timedelta(days=5), 'user_id': 'u1', 'product_id': 'p1', 'event_type': 'purchase', 'price': 100},
        {'event_time': now - timedelta(days=1), 'user_id': 'u1', 'product_id': 'p1', 'event_type': 'view', 'price': 100},
        
        # User 2: 1 view at T-8, 1 purchase in FUTURE (T+2)
        {'event_time': now - timedelta(days=8), 'user_id': 'u2', 'product_id': 'p2', 'event_type': 'view', 'price': 50},
        {'event_time': now + timedelta(days=2), 'user_id': 'u2', 'product_id': 'p2', 'event_type': 'purchase', 'price': 50},
    ]
    return pd.DataFrame(data), now

def test_feature_leakage_and_logic(mock_events):
    df, snapshot_time = mock_events
    
    # Generate features for 7 day horizon
    features = generate_user_features(df, snapshot_time, horizon_days=7)
    
    assert not features.empty
    
    # Convert features to dict for easy assert
    feat_dict = features.set_index('user_id').to_dict('index')
    
    # --- User 1 Asserts ---
    # Recency since last event: T-1 view -> 1 day
    assert pytest.approx(feat_dict['u1']['recency_days_since_last_event'], 0.1) == 1.0
    # Recency since last purchase: T-5 purchase -> 5 days
    assert pytest.approx(feat_dict['u1']['recency_days_since_last_purchase'], 0.1) == 5.0
    # Did they purchase in NEXT 7 days? No.
    assert feat_dict['u1']['label_purchased_next_7d'] == 0
    # Last 7 days purchase count = 1
    assert feat_dict['u1']['purchases_last_7_days'] == 1
    
    # --- User 2 Asserts ---
    # Recency since last event: T-8 view -> 8 days
    assert pytest.approx(feat_dict['u2']['recency_days_since_last_event'], 0.1) == 8.0
    # No purchase history -> default 999.0
    assert feat_dict['u2']['recency_days_since_last_purchase'] == 999.0
    # Did they purchase in NEXT 7 days? Yes (T+2)
    assert feat_dict['u2']['label_purchased_next_7d'] == 1
    # Last 7 days purchase count = 0 (future event shouldn't leak!)
    assert feat_dict['u2']['purchases_last_7_days'] == 0
    
def test_empty_history():
    df = pd.DataFrame(columns=['event_time', 'user_id', 'product_id', 'event_type', 'price'])
    snapshot = pd.to_datetime('2025-01-10 12:00:00')
    features = generate_user_features(df, snapshot, 7)
    assert features.empty
