"""
Phân cụm khách hàng bằng K-Means.
Features: RFM (Recency, Frequency, Monetary) + purchase_frequency
Kết quả lưu vào tbl_customer_segments và tbl_user_segments.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sqlalchemy.orm import Session
from sqlalchemy import text

from data.preprocessor import load_user_features

MODEL_PATH = Path("models/saved/kmeans.pkl")
SCALER_PATH = Path("models/saved/kmeans_scaler.pkl")

# Tên cụm mặc định (sẽ được gán theo đặc điểm sau khi train)
SEGMENT_NAMES = {
    0: "Khách hàng tiềm năng",
    1: "Khách hàng trung thành",
    2: "Khách hàng không hoạt động",
    3: "Khách hàng VIP",
}

FEATURES = ["total_orders", "total_spent", "avg_order_value",
            "days_since_last_order", "purchase_frequency"]


def _find_best_k(X_scaled: np.ndarray, k_range=range(2, 7)) -> int:
    """Tìm k tối ưu bằng Silhouette Score."""
    best_k, best_score = 2, -1
    for k in k_range:
        if len(X_scaled) < k:
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def train(db: Session, n_clusters: int = None) -> dict:
    """
    Huấn luyện K-Means, lưu model và cập nhật DB.
    n_clusters=None sẽ tự tìm k tối ưu.
    """
    df = load_user_features()
    if df.empty or len(df) < 4:
        return {"error": "Chưa đủ dữ liệu người dùng"}

    X = df[FEATURES].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = n_clusters or _find_best_k(X_scaled)
    k = min(k, len(df))  # không thể có nhiều cụm hơn số user

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels) if k > 1 else 0.0

    # Lưu model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(km, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Phân tích đặc điểm từng cụm để đặt tên
    df["cluster"] = labels
    cluster_stats = df.groupby("cluster")[FEATURES].mean()
    segment_labels = _label_segments(cluster_stats, k)

    # Cập nhật tbl_customer_segments
    db.execute(text("DELETE FROM tbl_user_segments"))
    db.execute(text("DELETE FROM tbl_customer_segments"))

    for cluster_id, seg_name in segment_labels.items():
        db.execute(
            text("INSERT INTO tbl_customer_segments (pk_segment_id, name, description) VALUES (:id, :name, :desc)"),
            {"id": cluster_id + 1, "name": seg_name, "desc": f"Cụm {cluster_id} - tự động phân loại bởi K-Means"},
        )

    # Gán user vào segment
    rows = [
        {"user_id": int(row.user_id), "segment_id": int(row.cluster) + 1}
        for row in df[["user_id", "cluster"]].itertuples()
    ]
    if rows:
        db.execute(
            text("INSERT INTO tbl_user_segments (fk_user_id, fk_segment_id) VALUES (:user_id, :segment_id)"),
            rows,
        )
    db.commit()

    return {
        "n_clusters": k,
        "silhouette_score": round(sil_score, 4),
        "segment_distribution": df["cluster"].value_counts().to_dict(),
        "segment_names": segment_labels,
    }


def _label_segments(stats: pd.DataFrame, k: int) -> dict:
    """Đặt tên cụm dựa trên đặc điểm RFM."""
    labels = {}
    for cid in range(k):
        if cid not in stats.index:
            labels[cid] = SEGMENT_NAMES.get(cid, f"Nhóm {cid}")
            continue
        row = stats.loc[cid]
        if row["total_spent"] > stats["total_spent"].quantile(0.75):
            labels[cid] = "Khách hàng VIP"
        elif row["days_since_last_order"] > stats["days_since_last_order"].quantile(0.75):
            labels[cid] = "Khách hàng không hoạt động"
        elif row["purchase_frequency"] > stats["purchase_frequency"].median():
            labels[cid] = "Khách hàng trung thành"
        else:
            labels[cid] = "Khách hàng tiềm năng"
    return labels


def predict_segment(user_id: int, db: Session) -> dict:
    """Lấy segment hiện tại của user từ DB."""
    result = db.execute(
        text("""
            SELECT cs.pk_segment_id, cs.name
            FROM tbl_user_segments us
            JOIN tbl_customer_segments cs ON cs.pk_segment_id = us.fk_segment_id
            WHERE us.fk_user_id = :uid
        """),
        {"uid": user_id},
    ).fetchone()

    if not result:
        return {"user_id": user_id, "segment": None}
    return {"user_id": user_id, "segment_id": result.pk_segment_id, "segment_name": result.name}
