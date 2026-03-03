import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import pickle
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_classifier():
    processed_dir = Path(__file__).parent.parent / config['data']['processed_dir']
    test_path = processed_dir / "classifier_test_set.parquet"
    model_path = processed_dir / "purchase_classifier.joblib"
    
    if not test_path.exists() or not model_path.exists():
        logger.error("Classifier test data or model missing.")
        return
        
    df = pd.read_parquet(test_path)
    model = joblib.load(model_path)
    
    features_cols = [c for c in df.columns if c not in ('snapshot_date', 'user_id', 'label_purchased_next_7d')]
    X_test = df[features_cols]
    y_test = df['label_purchased_next_7d']
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    with open(reports_dir / "classifier_metrics.txt", "w") as f:
        f.write("=== Purchase Prediction Classifier ===\n")
        f.write(f"ROC-AUC:   {roc_auc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        
    logger.info(f"Classifier Eval - ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}")

def evaluate_recommender():
    processed_dir = Path(__file__).parent.parent / config['data']['processed_dir']
    test_path = processed_dir / "recommender_test_set.parquet"
    model_path = processed_dir / "recommender_model.pkl"
    
    if not test_path.exists() or not model_path.exists():
        logger.error("Recommender test data or model missing.")
        return
        
    test_df = pd.read_parquet(test_path)
    # Ground truth: Products actually interacted with in test period
    # Focus on purchases for hit rate evaluation
    purchases = test_df[test_df['event_type'] == 'purchase']
    user_ground_truth = purchases.groupby('user_id')['product_id'].apply(set).to_dict()
    
    if not user_ground_truth:
         logger.warning("No purchases in test set to evaluate recommender.")
         return
         
    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
        
    model = artifacts['model']
    u2i, i2u = artifacts['user_to_idx'], artifacts['idx_to_user']
    item2idx = artifacts['item_to_idx']
    idx2item = artifacts['idx_to_item']
    user_items = artifacts['user_item_csr']
    
    k = config['model_recommender']['recommend_k']
    
    hits = 0
    total_users = 0
    map_k = 0.0
    
    for user, actual_items in user_ground_truth.items():
        if user not in u2i:
            continue
            
        uidx = u2i[user]
        # Recommend
        ids, scores = model.recommend(uidx, user_items[uidx], N=k, filter_already_liked_items=False)
        rec_items = [idx2item[i] for i in ids]
        
        # HitRate
        is_hit = int(len(set(rec_items).intersection(actual_items)) > 0)
        hits += is_hit
        
        # MAP@K
        ap = 0.0
        hits_in_k = 0
        for i, item in enumerate(rec_items):
            if item in actual_items:
                hits_in_k += 1
                ap += hits_in_k / (i + 1.0)
                
        # Normalize AP
        map_k += ap / min(len(actual_items), k)
        total_users += 1
        
    if total_users == 0:
        logger.warning("No valid users in test set found in training set.")
        return
        
    hit_rate = hits / total_users
    map_k = map_k / total_users
    
    logger.info(f"Recommender Eval - HitRate@{k}: {hit_rate:.4f}, MAP@{k}: {map_k:.4f}")
    
    reports_dir = Path(__file__).parent.parent / "reports"
    with open(reports_dir / "recommender_metrics.txt", "w") as f:
        f.write("=== Recommender Configuration ===\n")
        f.write(f"HitRate@{k}: {hit_rate:.4f}\n")
        f.write(f"MAP@{k}:     {map_k:.4f}\n")
        f.write(f"Evaluated on {total_users} users with purchases in test window.\n")

if __name__ == "__main__":
    evaluate_classifier()
    evaluate_recommender()
    print("Evaluation Complete. Check /reports/ directory.")
