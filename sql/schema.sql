-- events_raw table
CREATE TABLE IF NOT EXISTS events_raw (
    event_time TIMESTAMP NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    product_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(20) NOT NULL,
    price FLOAT,
    category_id VARCHAR(50),
    brand VARCHAR(50),
    session_id VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_events_raw_user ON events_raw(user_id);
CREATE INDEX IF NOT EXISTS idx_events_raw_product ON events_raw(product_id);
CREATE INDEX IF NOT EXISTS idx_events_raw_time ON events_raw(event_time);

-- users_dim table
CREATE TABLE IF NOT EXISTS users_dim (
    user_id VARCHAR(50) PRIMARY KEY,
    first_seen TIMESTAMP
);

-- products_dim table
CREATE TABLE IF NOT EXISTS products_dim (
    product_id VARCHAR(50) PRIMARY KEY,
    category_id VARCHAR(50),
    brand VARCHAR(50),
    avg_price FLOAT
);

-- features_user_daily table
CREATE TABLE IF NOT EXISTS features_user_daily (
    snapshot_date DATE NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    recency_days_since_last_event FLOAT,
    recency_days_since_last_purchase FLOAT,
    events_last_7_days INT,
    events_last_30_days INT,
    purchases_last_7_days INT,
    purchases_last_30_days INT,
    carts_last_7_days INT,
    carts_last_30_days INT,
    views_last_7_days INT,
    views_last_30_days INT,
    conversion_rate_7d FLOAT,
    cart_rate_7d FLOAT,
    avg_purchase_price_30d FLOAT,
    unique_products_viewed_30d INT,
    label_purchased_next_7d INT,
    PRIMARY KEY (snapshot_date, user_id)
);

-- predictions table
CREATE TABLE IF NOT EXISTS predictions (
    prediction_time TIMESTAMP NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    prob_purchase FLOAT NOT NULL
);

-- recommendations table
CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_time TIMESTAMP NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    product_id VARCHAR(50) NOT NULL,
    rank INT NOT NULL,
    score FLOAT
);
