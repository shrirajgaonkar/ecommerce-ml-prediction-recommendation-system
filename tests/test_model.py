import pytest
import numpy as np
from scipy.sparse import csr_matrix
import implicit

def test_recommender_shape_and_logic():
    # Toy Matrix: 4 items x 3 users
    # implicit expects USER x ITEM matrix
    # So we build item-user then transpose before fit
    
    row = np.array([0, 1, 2, 3, 0, 2])
    col = np.array([0, 0, 0, 1, 2, 2])
    data = np.array([6.0, 3.0, 1.0, 6.0, 3.0, 6.0])
    
    item_user_csr = csr_matrix((data, (row, col)), shape=(4, 3))
    
    # Transpose because implicit expects users x items
    user_item_csr = item_user_csr.T.tocsr()
    
    model = implicit.als.AlternatingLeastSquares(
        factors=2,
        regularization=0.1,
        iterations=10
    )
    
    model.fit(user_item_csr)
    
    # Now dimensions are correct
    assert model.user_factors.shape[0] == 3   # 3 users
    assert model.item_factors.shape[0] == 4   # 4 items
    
    # Recommend for User 0
    ids, scores = model.recommend(
        0,
        user_item_csr[0],
        N=2,
        filter_already_liked_items=False
    )
    
    assert len(ids) == 2
    assert len(set(ids)) == 2  # no duplicates


def test_predict_proba_shape():
    from sklearn.linear_model import LogisticRegression
    
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = LogisticRegression()
    model.fit(X, y)
    
    probas = model.predict_proba(X)
    
    assert probas.shape == (100, 2)
    assert np.all((probas >= 0) & (probas <= 1))
    assert np.allclose(np.sum(probas, axis=1), 1.0)