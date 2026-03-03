# E-Commerce Purchase Prediction & Recommendation System

An end-to-end Machine Learning system for predicting user purchase intent and generating personalized product recommendations using behavioral e-commerce event data.

---

## 🚀 Project Overview

This project simulates a real-world production ML pipeline that:

- Processes 50,000+ user interaction events
- Engineers leakage-free behavioral features
- Trains a purchase prediction classifier
- Builds a collaborative filtering recommendation engine
- Evaluates model performance using industry metrics
- Serves predictions through a FastAPI production API
- Includes automated unit testing

The system is modular, config-driven, and production structured.

---

## 🏗️ System Architecture

Synthetic Data Generation  
↓  
SQLite Database (events_raw, features_user_daily, etc.)  
↓  
Feature Engineering (Time-based snapshots)  
↓  
Time-Aware Train/Test Split  
↓  
Purchase Prediction Model (HistGradientBoosting)  
↓  
ALS Collaborative Filtering Recommender  
↓  
Evaluation Reports (ROC-AUC, F1, MAP@K, HitRate@K)  
↓  
FastAPI ML Serving Layer  

---

## 📊 Core Components

### 1️⃣ Data Layer

- Synthetic generation of 55,000+ e-commerce interaction events
- Indexed SQL schema
- User and product dimension tables
- SQLite-backed pipeline

Tables include:

- `events_raw`
- `users_dim`
- `products_dim`
- `features_user_daily`
- `predictions`
- `recommendations`

---

### 2️⃣ Feature Engineering

Time-aware snapshot-based features:

- Recency (days since last event / purchase)
- Events in last 7 / 30 days
- Purchases / Cart / View counts
- Conversion rate (7-day)
- Cart rate (7-day)
- Avg purchase price (30-day)
- Unique products viewed (30-day)
- Strict history/future split to prevent data leakage

---

### 3️⃣ Purchase Prediction Model

Model:
- `HistGradientBoostingClassifier`

Pipeline:
- Median imputation
- Standard scaling
- Time-based train/test split

Evaluation metrics:
- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix

Designed to simulate real-world purchase intent modeling.

---

### 4️⃣ Recommendation Engine

Model:
- ALS (Alternating Least Squares) using `implicit` library

Interaction weights:
- View = 1
- Add to cart = 3
- Purchase = 6

Sparse matrix factorization approach.

Evaluation metrics:
- HitRate@K
- MAP@K

Collaborative filtering optimized for behavioral signals.

---

### 5️⃣ FastAPI ML Serving Layer

Production-ready REST API.

Endpoints:

| Method | Endpoint     | Description |
|--------|-------------|------------|
| GET    | `/health`   | Check model status |
| POST   | `/predict`  | Predict purchase probability |
| POST   | `/recommend`| Generate top-N product recommendations |

Swagger UI available at:

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Unit Testing

Implemented with Pytest.

Test coverage includes:

- Synthetic data validation
- Feature leakage prevention
- Recommender matrix correctness
- Model probability output validation

Run tests:

```bash
pytest
```

---

## ⚙️ Configuration-Driven Design

All system parameters controlled via `config.yaml`:

- Snapshot horizon
- Train/test cutoff window
- Recommender factors
- Regularization strength
- Interaction weights
- API configuration

Enables experimentation without modifying source code.

---

## 🛠️ How To Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Run Full ML Pipeline

```bash
python -m src.ingest
python -m src.features
python -m src.train_classifier
python -m src.train_recommender
python -m src.evaluate
```

This will:

- Generate synthetic data
- Build features
- Train models
- Save artifacts
- Generate evaluation reports

---

### 3️⃣ Start API Server

```bash
uvicorn src.api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 📈 Sample API Output

### Purchase Prediction

```json
{
  "user_id": "user_0001",
  "purchase_probability": 0.27
}
```

### Recommendation

```json
{
  "user_id": "user_0001",
  "recommended_product_ids": [
    "prod_0123",
    "prod_0045",
    "prod_0345"
  ]
}
```

---

## 📂 Project Structure

```
ecomm_ml_system/
│
├── src/
│   ├── ingest.py
│   ├── features.py
│   ├── train_classifier.py
│   ├── train_recommender.py
│   ├── evaluate.py
│   ├── api.py
│   ├── db.py
│   └── utils.py
│
├── sql/
│   └── schema.sql
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_model.py
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 🎯 Engineering Highlights

- Time-aware train/test split
- Leakage-free feature engineering
- Sparse matrix collaborative filtering
- Config-driven experimentation
- Modular ML architecture
- Production API serving
- Automated evaluation
- Unit test coverage

---

## 💡 What This Demonstrates

This project demonstrates:

- Applied Machine Learning
- Recommender Systems
- ML Engineering best practices
- Feature pipeline design
- SQL integration
- API deployment
- Testing discipline
- Production-ready architecture

---

## 🚀 Future Enhancements

- Docker containerization
- Cloud deployment (AWS / GCP / Azure)
- Real-world dataset integration
- Monitoring & model drift detection
- Online A/B testing simulation
- Feature store integration

---

## 👨‍💻 Author

Developed as a full-stack Machine Learning Engineering project demonstrating end-to-end production ML capabilities.