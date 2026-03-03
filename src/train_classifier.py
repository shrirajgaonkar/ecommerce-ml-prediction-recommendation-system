import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from src.utils import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_classifier():
    processed_dir = Path(__file__).parent.parent / config['data']['processed_dir']
    features_path = processed_dir / "features_user_daily.parquet"
    
    if not features_path.exists():
        logger.error("Features not found. Please run feature engineering first.")
        return
        
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} feature rows for training.")
    
    cutoff_days = config['model_classifier']['cutoff_days']
    max_date = pd.to_datetime(df['snapshot_date']).max()
    cutoff_date = max_date - pd.Timedelta(days=cutoff_days)
    
    # Time-based split
    train_df = df[pd.to_datetime(df['snapshot_date']) < cutoff_date]
    test_df = df[pd.to_datetime(df['snapshot_date']) >= cutoff_date]
    
    logger.info(f"Train samples: {len(train_df)}. Test samples: {len(test_df)}")
    
    features_cols = [c for c in df.columns if c not in ('snapshot_date', 'user_id', 'label_purchased_next_7d')]
    target_col = 'label_purchased_next_7d'
    
    X_train, y_train = train_df[features_cols], train_df[target_col]
    X_test, y_test = test_df[features_cols], test_df[target_col]
    
    # Assume all features currently are numerical
    numeric_features = features_cols
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ], remainder='drop')
        
    model_type = config['model_classifier'].get('model_type', 'hist_gradient_boosting')
    
    if model_type == 'hist_gradient_boosting':
        classifier = HistGradientBoostingClassifier(
            max_iter=100, 
            learning_rate=0.1, 
            random_state=config['model_classifier']['random_state']
        )
    else:
        classifier = LogisticRegression(
            max_iter=1000, 
            random_state=config['model_classifier']['random_state']
        )
        
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', classifier)])
                               
    logger.info(f"Training {model_type} pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Save model
    model_path = processed_dir / "purchase_classifier.joblib"
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved classifier model to {model_path}.")
    
    # Write test data out for later evaluation
    test_df.to_parquet(processed_dir / "classifier_test_set.parquet", index=False)
    
if __name__ == "__main__":
    train_classifier()
