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

FEATURES = ["quantity", "weight_gram", "product_id_enc", "avg_days_between"]


def train(db: Session) -> dict:
    """Huấn luyện Random Forest, lưu model và tạo dự đoán cho tất cả user."""
    df = load_repurchase_data()
    if df.empty or len(df) < 20:
        return {"error": "Chưa đủ dữ liệu mua lại để huấn luyện"}

    df = df.fillna({"weight_gram": df["weight_gram"].median()})

    # Thêm avg_days_between: trung bình chu kỳ mua lại theo từng user-product
    avg_cycle = (
        df.groupby(["user_id", "product_id"])["days_between"]
        .mean()
        .reset_index()
        .rename(columns={"days_between": "avg_days_between"})
    )
    df = df.merge(avg_cycle, on=["user_id", "product_id"], how="left")

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

    # Confidence dựa trên R² — phản ánh độ tin cậy thực tế của model
    # Clip về [0.3, 0.95] để tránh giá trị cực đoan
    model_confidence = float(np.clip(r2, 0.3, 0.95))

    # Lưu model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    # Tạo dự đoán cho tất cả user-product pairs
    _generate_predictions(model, le, db, model_confidence)

    return {"mae_days": round(mae, 2), "r2_score": round(r2, 4), "n_samples": len(df)}


def _generate_predictions(model, le: LabelEncoder, db: Session, model_confidence: float = 0.75):
    """Tạo dự đoán mua lại cho từng user-product và lưu vào DB."""
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

    # Tính avg_days_between từ lịch sử thực tế
    history_query = """
        SELECT o.fk_user_id AS user_id, oi.fk_product_id AS product_id, o.created_at AS order_date
        FROM tbl_order_items oi
        JOIN tbl_orders o ON o.pk_order_id = oi.fk_order_id AND o.order_status = 'delivered'
        JOIN tbl_products p ON p.pk_product_id = oi.fk_product_id AND p.is_consumable = 1
        ORDER BY o.fk_user_id, oi.fk_product_id, o.created_at
    """
    hist = fetch_df(history_query)
    if not hist.empty:
        hist["order_date"] = pd.to_datetime(hist["order_date"])
        hist = hist.sort_values(["user_id", "product_id", "order_date"])
        hist["prev_date"] = hist.groupby(["user_id", "product_id"])["order_date"].shift(1)
        hist = hist.dropna(subset=["prev_date"])
        hist["days_between"] = (hist["order_date"] - hist["prev_date"]).dt.days
        avg_cycle = (
            hist.groupby(["user_id", "product_id"])["days_between"]
            .mean()
            .reset_index()
            .rename(columns={"days_between": "avg_days_between"})
        )
        df = df.merge(avg_cycle, on=["user_id", "product_id"], how="left")
    else:
        df["avg_days_between"] = 30.0  # fallback

    df["avg_days_between"] = df["avg_days_between"].fillna(30.0)

    df["product_id_enc"] = le.transform(df["product_id"])
    X = df[["avg_quantity", "weight_gram", "product_id_enc", "avg_days_between"]].values
    predicted_days = model.predict(X).clip(min=1)

    today = pd.Timestamp.today().normalize()
    df["last_purchase"] = pd.to_datetime(df["last_purchase"])

    # Tính predicted_date = last_purchase + predicted_days
    # Nếu kết quả đã qua (last_purchase quá cũ), tính lại từ hôm nay
    # dựa trên phần dư của chu kỳ: days_since_last % predicted_days
    raw_dates = [
        pd.to_datetime(last) + timedelta(days=int(d))
        for last, d in zip(df["last_purchase"], predicted_days)
    ]
    adjusted_dates = []
    for raw_date, last, d in zip(raw_dates, df["last_purchase"], predicted_days):
        if raw_date.date() >= today.date():
            adjusted_dates.append(raw_date.date())
        else:
            # Tính số ngày đã qua kể từ last_purchase
            days_since = (today - pd.to_datetime(last)).days
            # Số ngày còn lại trong chu kỳ hiện tại
            remainder = int(d) - (days_since % int(d))
            adjusted_dates.append((today + timedelta(days=remainder)).date())

    df["predicted_date"] = adjusted_dates
    df["confidence"] = round(model_confidence, 4)

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


def send_repurchase_notifications(db: Session, days_ahead: int = 3) -> dict:
    """
    Tạo thông báo nhắc mua lại cho các prediction sắp đến hạn chưa được thông báo.
    Chạy hàng ngày qua scheduler.
    """
    rows = db.execute(
        text("""
            SELECT rp.pk_pred_id, rp.fk_user_id, rp.fk_product_id,
                   rp.predicted_date, rp.confidence, p.name AS product_name
            FROM tbl_repurchase_predictions rp
            JOIN tbl_products p ON p.pk_product_id = rp.fk_product_id
            WHERE rp.notified = 0
              AND rp.predicted_date BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL :days DAY)
        """),
        {"days": days_ahead},
    ).fetchall()

    if not rows:
        return {"notifications_sent": 0}

    notif_rows = [
        {
            "user_id": r.fk_user_id,
            "type": "repurchase_reminder",
            "title": "Nhắc mua lại sản phẩm",
            "message": (
                f"Sản phẩm '{r.product_name}' của bạn sắp hết. "
                f"Dự kiến bạn cần mua lại vào ngày {r.predicted_date}."
            ),
            "ref_id": r.fk_product_id,
        }
        for r in rows
    ]

    db.execute(
        text("""
            INSERT INTO tbl_notifications (fk_user_id, type, title, message, ref_id)
            VALUES (:user_id, :type, :title, :message, :ref_id)
        """),
        notif_rows,
    )

    pred_ids = [r.pk_pred_id for r in rows]
    db.execute(
        text("UPDATE tbl_repurchase_predictions SET notified = 1 WHERE pk_pred_id IN :ids"),
        {"ids": tuple(pred_ids)},
    )
    db.commit()

    return {"notifications_sent": len(notif_rows)}
