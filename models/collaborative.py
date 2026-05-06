"""
Collaborative Filtering dựa trên hành vi người dùng.
Dùng TruncatedSVD (Matrix Factorization) từ scikit-learn - không cần build C++.
Fallback về Item-based cosine similarity khi user mới (cold start).
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from data.preprocessor import load_behavior_logs, build_interaction_matrix

MODEL_PATH = Path("models/saved/collaborative_svd.pkl")
MATRIX_PATH = Path("models/saved/interaction_matrix.pkl")
MIN_INTERACTIONS = 5  # Số sản phẩm đã tương tác tối thiểu để dùng SVD


def train(save: bool = True) -> dict:
    """Huấn luyện TruncatedSVD. Trả về metrics đánh giá."""
    df = load_behavior_logs()
    if df.empty:
        return {"error": "Không có dữ liệu hành vi"}

    matrix = build_interaction_matrix(df)

    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return {"error": "Chưa đủ dữ liệu để huấn luyện"}

    n_components = min(50, matrix.shape[0] - 1, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    # Đánh giá bằng cross-validation thủ công
    X = matrix.values
    rmse_scores = []
    kf = KFold(n_splits=min(3, matrix.shape[0]), shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_train = X.copy()
        X_train[test_idx] = 0
        svd_cv = TruncatedSVD(n_components=n_components, random_state=42)
        X_approx = svd_cv.fit_transform(X_train)
        X_reconstructed = X_approx @ svd_cv.components_
        mask = X[test_idx] > 0
        if mask.any():
            rmse = np.sqrt(mean_squared_error(
                X[test_idx][mask],
                X_reconstructed[test_idx][mask]
            ))
            rmse_scores.append(rmse)

    # Train trên toàn bộ dữ liệu
    svd.fit(matrix.values)

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(svd, f)
        with open(MATRIX_PATH, "wb") as f:
            pickle.dump(matrix, f)

    return {
        "rmse": round(float(np.mean(rmse_scores)), 4) if rmse_scores else None,
        "n_components": n_components,
        "n_users": matrix.shape[0],
        "n_products": matrix.shape[1],
    }


def _load_model() -> TruncatedSVD:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model chưa được huấn luyện. Gọi POST /train/collaborative trước.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _load_matrix() -> pd.DataFrame:
    if not MATRIX_PATH.exists():
        raise FileNotFoundError("Interaction matrix chưa tồn tại.")
    with open(MATRIX_PATH, "rb") as f:
        return pickle.load(f)


def predict_for_user(user_id: int, top_n: int = 10) -> list[dict]:
    """
    Gợi ý top_n sản phẩm cho user_id.
    - Nếu user có đủ lịch sử: dùng SVD reconstruction
    - Nếu user mới (cold start): dùng item-based cosine similarity
    """
    matrix = _load_matrix()
    svd = _load_model()

    if user_id in matrix.index:
        user_row = matrix.loc[user_id].values
        n_interacted = int((user_row > 0).sum())
        unseen_mask = user_row == 0
        unseen_ids = matrix.columns[unseen_mask].tolist()
    else:
        user_row = np.zeros(matrix.shape[1])
        n_interacted = 0
        unseen_ids = matrix.columns.tolist()

    if not unseen_ids:
        return []

    if n_interacted >= MIN_INTERACTIONS:
        # SVD: reconstruct toàn bộ vector rồi lấy score của unseen
        user_latent = svd.transform(user_row.reshape(1, -1))
        reconstructed = (user_latent @ svd.components_).flatten()
        product_index = {pid: i for i, pid in enumerate(matrix.columns)}
        preds = [
            {"product_id": int(pid), "score": round(float(reconstructed[product_index[pid]]), 4)}
            for pid in unseen_ids
        ]
    else:
        preds = _item_based_fallback(user_id, unseen_ids, matrix)

    preds.sort(key=lambda x: x["score"], reverse=True)
    return preds[:top_n]


def _item_based_fallback(user_id: int, unseen: list, matrix: pd.DataFrame) -> list[dict]:
    """Cosine similarity giữa các sản phẩm user đã tương tác và sản phẩm chưa xem."""
    if user_id not in matrix.index:
        return [{"product_id": int(pid), "score": 0.0} for pid in unseen]

    user_row = matrix.loc[user_id]
    seen = user_row[user_row > 0].index.tolist()
    if not seen:
        return [{"product_id": int(pid), "score": 0.0} for pid in unseen]

    item_matrix = matrix.T  # shape: (products, users)
    sim = cosine_similarity(
        item_matrix.loc[unseen].values,
        item_matrix.loc[seen].values,
    )
    seen_scores = user_row[seen].values
    scores = sim.dot(seen_scores) / (sim.sum(axis=1) + 1e-9)

    return [
        {"product_id": int(pid), "score": round(float(s), 4)}
        for pid, s in zip(unseen, scores)
    ]
