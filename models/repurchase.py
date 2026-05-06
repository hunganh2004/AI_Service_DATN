"""
Dự đoán thời điểm mua lại sản phẩm tiêu hao (is_consumable=1).
Dùng Random Forest Regressor để dự đoán số ngày đến lần mua tiếp theo.
Kết quả lưu vào tbl_repurchase_predictions.
"""
import pickle
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sqlalchemy.orm import Session
from sqlalchemy import text

from data.preprocessor import load_repurchase_data
from database import fetch_df

MODEL_PATH = Path("models/saved/repurchase_rf.pkl")
ENCODER_PATH = Path("models/saved/repurchase_encoder.pkl")

FEATURES = ["quantity", "weight_gram", "product_id_enc"]


def train(db: Session) -> dict:
    """Huấn luyện Random Forest, lưu model và tạo dự đoán cho tất cả user."""
    df = load_repurchase_data()
    if df.empty or len(df) < 20:
        return {"error": "Chưa đủ dữ liệu mua lại để huấn luyện"}

    df = df.fillna({"weight_gram": df["weight_gram"].median()})

    le = LabelEncoder()
    df["product_id_enc"] = le.fit_transform(df["product_id"])

    X = df[FEATURES].values
    y = df["days_between"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Lưu model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    # Tạo dự đoán cho tất cả user-product pairs
    _generate_predictions(model, le, db)

    return {"mae_days": round(mae, 2), "r2_score": round(r2, 4), "n_samples": len(df)}


def _generate_predictions(model, le: LabelEncoder, db: Session):
    """Tạo dự đoán mua lại cho từng user-product và lưu vào DB."""
    # Lấy lần mua cuối cùng của mỗi user-product tiêu hao
    query = """
        SELECT
            o.fk_user_id        AS user_id,
            oi.fk_product_id    AS product_id,
            MAX(o.created_at)   AS last_purchase,
            AVG(oi.quantity)    AS avg_quantity,
            p.weight_gram
        FROM tbl_order_items oi
        JOIN tbl_orders o   ON o.pk_order_id = oi.fk_order_id
                           AND o.order_status = 'delivered'
        JOIN tbl_products p ON p.pk_product_id = oi.fk_product_id
                           AND p.is_consumable = 1
        GROUP BY o.fk_user_id, oi.fk_product_id, p.weight_gram
    """
    df = fetch_df(query)
    if df.empty:
        return

    df["weight_gram"] = df["weight_gram"].fillna(df["weight_gram"].median())
    df["avg_quantity"] = df["avg_quantity"].fillna(1)

    # Encode product_id (chỉ encode những id đã biết)
    known_ids = set(le.classes_)
    df = df[df["product_id"].isin(known_ids)].copy()
    if df.empty:
        return

    df["product_id_enc"] = le.transform(df["product_id"])
    X = df[["avg_quantity", "weight_gram", "product_id_enc"]].values
    predicted_days = model.predict(X).clip(min=1)

    df["predicted_date"] = [
        (pd.to_datetime(last) + timedelta(days=int(d))).date()
        for last, d in zip(df["last_purchase"], predicted_days)
    ]
    # Confidence đơn giản: 1 - normalized MAE (placeholder)
    df["confidence"] = 0.75

    # Xoá dự đoán cũ, insert mới
    db.execute(text("DELETE FROM tbl_repurchase_predictions"))
    rows = df[["user_id", "product_id", "predicted_date", "confidence"]].to_dict("records")
    if rows:
        db.execute(
            text("""
                INSERT INTO tbl_repurchase_predictions
                    (fk_user_id, fk_product_id, predicted_date, confidence)
                VALUES (:user_id, :product_id, :predicted_date, :confidence)
            """),
            rows,
        )
    db.commit()


def get_upcoming_repurchases(user_id: int, db: Session, days_ahead: int = 7) -> list[dict]:
    """Lấy danh sách sản phẩm user cần mua lại trong days_ahead ngày tới."""
    result = db.execute(
        text("""
            SELECT rp.fk_product_id AS product_id, rp.predicted_date, rp.confidence,
                   p.name AS product_name
            FROM tbl_repurchase_predictions rp
            JOIN tbl_products p ON p.pk_product_id = rp.fk_product_id
            WHERE rp.fk_user_id = :uid
              AND rp.predicted_date BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL :days DAY)
            ORDER BY rp.predicted_date
        """),
        {"uid": user_id, "days": days_ahead},
    ).fetchall()

    return [
        {
            "product_id": row.product_id,
            "product_name": row.product_name,
            "predicted_date": str(row.predicted_date),
            "confidence": row.confidence,
        }
        for row in result
    ]
