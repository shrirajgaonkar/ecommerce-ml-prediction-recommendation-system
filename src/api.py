from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from src.utils import config
from src.db import execute_query

app = FastAPI(title="E-Commerce ML API")

# Load Models (cached)
models_cache = {}

def load_models():
    if not models_cache:
        processed_dir = Path(__file__).parent.parent / config['data']['processed_dir']
        
        try:
            models_cache['classifier'] = joblib.load(processed_dir / "purchase_classifier.joblib")
            
            with open(processed_dir / "recommender_model.pkl", 'rb') as f:
                models_cache['recommender'] = pickle.load(f)
        except Exception as e:
            print(f"Error loading models: {e}")

class PredictRequest(BaseModel):
    user_id: str
    
class PredictResponse(BaseModel):
    user_id: str
    purchase_probability: float
    
class RecommendRequest(BaseModel):
    user_id: str
    n_recommendations: int = 10
    
class RecommendResponse(BaseModel):
    user_id: str
    recommended_product_ids: List[str]

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/health")
def read_health():
    if 'classifier' in models_cache and 'recommender' in models_cache:
        return {"status": "healthy"}
    return {"status": "models_missing"}

@app.post("/predict", response_model=PredictResponse)
def predict_purchase(req: PredictRequest):
    if 'classifier' not in models_cache:
        raise HTTPException(status_code=503, detail="Classifier not loaded.")
        
    # Get latest features for user
    query = f"SELECT * FROM features_user_daily WHERE user_id = '{req.user_id}' ORDER BY snapshot_date DESC LIMIT 1"
    df = execute_query(query, as_df=True)
    
    if df.empty:
        # Default cold start logic
        return PredictResponse(user_id=req.user_id, purchase_probability=0.01)
        
    features_cols = [c for c in df.columns if c not in ('snapshot_date', 'user_id', 'label_purchased_next_7d')]
    X = df[features_cols]
    
    prob = models_cache['classifier'].predict_proba(X)[0][1]
    
    return PredictResponse(user_id=req.user_id, purchase_probability=prob)

@app.post("/recommend", response_model=RecommendResponse)
def recommend_items(req: RecommendRequest):
    if 'recommender' not in models_cache:
        raise HTTPException(status_code=503, detail="Recommender not loaded.")
        
    artifacts = models_cache['recommender']
    model = artifacts['model']
    u2i = artifacts['user_to_idx']
    idx2item = artifacts['idx_to_item']
    user_items = artifacts['user_item_csr']
    
    if req.user_id not in u2i:
        # Cold start fallback - return top popular (pseudo baseline placeholder)
        default_recs = ["prod_0001", "prod_0002", "prod_0003"][:req.n_recommendations]
        return RecommendResponse(user_id=req.user_id, recommended_product_ids=default_recs)
        
    uidx = u2i[req.user_id]
    ids, _ = model.recommend(uidx, user_items[uidx], N=req.n_recommendations, filter_already_liked_items=False)
    
    recs = [idx2item[i] for i in ids]
    
    return RecommendResponse(user_id=req.user_id, recommended_product_ids=recs)

