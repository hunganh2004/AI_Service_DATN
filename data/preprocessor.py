"""
Tiền xử lý dữ liệu từ tbl_user_behavior_logs và các bảng liên quan.
Cung cấp các hàm load dữ liệu sẵn sàng cho từng mô hình AI.
"""
import pandas as pd
import numpy as np
from database import fetch_df

# Trọng số điểm cho từng hành vi người dùng
# Chỉ bao gồm các action có trong ENUM của tbl_user_behavior_logs
ACTION_WEIGHTS = {
    "view": 1,
    "search": 1,
    "wishlist": 2,
    "add_to_cart": 3,
    "remove_from_cart": -1,
    "purchase": 5,
}


def load_behavior_logs(min_date: str = None) -> pd.DataFrame:
    """
    Load toàn bộ hành vi người dùng (chỉ user đã đăng nhập).
    Trả về DataFrame với các cột: user_id, product_id, action, duration_sec, created_at
    """
    where = "WHERE fk_user_id IS NOT NULL AND fk_product_id IS NOT NULL"
    if min_date:
        where += f" AND created_at >= '{min_date}'"

    query = f"""
        SELECT
            fk_user_id   AS user_id,
            fk_product_id AS product_id,
            action,
            COALESCE(duration_sec, 0) AS duration_sec,
            created_at
        FROM tbl_user_behavior_logs
        {where}
        ORDER BY created_at
    """
    return fetch_df(query)


def build_interaction_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính điểm tương tác user-product từ hành vi.
    Trả về pivot table: rows=user_id, cols=product_id, values=score
    """
    df = df.copy()
    df["weight"] = df["action"].map(ACTION_WEIGHTS).fillna(0)

    # Bonus nhỏ cho thời gian xem dài (mỗi 30 giây = +0.1 điểm, tối đa +1)
    df["time_bonus"] = (df["duration_sec"] / 30).clip(upper=10) * 0.1

    df["score"] = df["weight"] + df["time_bonus"]

    # Tổng hợp điểm theo cặp user-product
    agg = (
        df.groupby(["user_id", "product_id"])["score"]
        .sum()
        .reset_index()
    )

    # Clip điểm âm về 0
    agg["score"] = agg["score"].clip(lower=0)

    pivot = agg.pivot(index="user_id", columns="product_id", values="score").fillna(0)
    return pivot


def load_order_items() -> pd.DataFrame:
    """
    Load lịch sử mua hàng theo đơn hàng (dùng cho Association Rules).
    Trả về DataFrame: order_id, product_id
    """
    query = """
        SELECT
            oi.fk_order_id  AS order_id,
            oi.fk_product_id AS product_id
        FROM tbl_order_items oi
        JOIN tbl_orders o ON o.pk_order_id = oi.fk_order_id
        WHERE o.order_status = 'delivered'
    """
    return fetch_df(query)


def load_user_features() -> pd.DataFrame:
    """
    Xây dựng feature vector cho từng user (dùng cho Clustering & Regression).
    Features: total_orders, total_spent, avg_order_value, days_since_last_order,
              favorite_pet_type (encoded), purchase_frequency (orders/month)
    """
    query = """
        SELECT
            u.pk_user_id                                        AS user_id,
            COALESCE(s.total_orders, 0)                         AS total_orders,
            COALESCE(s.total_spent, 0)                          AS total_spent,
            CASE WHEN COALESCE(s.total_orders, 0) > 0
                 THEN s.total_spent / s.total_orders ELSE 0
            END                                                 AS avg_order_value,
            COALESCE(
                DATEDIFF(NOW(), s.last_order_at), 9999
            )                                                   AS days_since_last_order,
            COALESCE(
                DATEDIFF(NOW(), u.created_at), 1
            )                                                   AS account_age_days
        FROM tbl_users u
        LEFT JOIN v_user_purchase_summary s ON s.pk_user_id = u.pk_user_id
        WHERE u.role = 'customer' AND u.is_active = 1
    """
    df = fetch_df(query)

    # purchase_frequency: đơn/tháng
    df["purchase_frequency"] = (
        df["total_orders"] / (df["account_age_days"] / 30).clip(lower=1)
    )

    return df


def load_repurchase_data() -> pd.DataFrame:
    """
    Dữ liệu cho mô hình dự đoán mua lại sản phẩm tiêu hao.
    Chỉ lấy các sản phẩm is_consumable=1 và user đã mua ít nhất 2 lần.
    Trả về: user_id, product_id, days_between (target), features
    """
    query = """
        SELECT
            oi.fk_product_id                        AS product_id,
            o.fk_user_id                            AS user_id,
            o.created_at                            AS order_date,
            oi.quantity,
            p.weight_gram
        FROM tbl_order_items oi
        JOIN tbl_orders o   ON o.pk_order_id = oi.fk_order_id
                           AND o.order_status = 'delivered'
        JOIN tbl_products p ON p.pk_product_id = oi.fk_product_id
                           AND p.is_consumable = 1
        ORDER BY o.fk_user_id, oi.fk_product_id, o.created_at
    """
    df = fetch_df(query)
    if df.empty:
        return df

    df["order_date"] = pd.to_datetime(df["order_date"])

    # Tính khoảng cách ngày giữa 2 lần mua liên tiếp cùng sản phẩm
    df = df.sort_values(["user_id", "product_id", "order_date"])
    df["prev_date"] = df.groupby(["user_id", "product_id"])["order_date"].shift(1)
    df = df.dropna(subset=["prev_date"])
    df["days_between"] = (df["order_date"] - df["prev_date"]).dt.days

    # Loại bỏ outlier (< 1 ngày hoặc > 365 ngày)
    df = df[(df["days_between"] >= 1) & (df["days_between"] <= 365)]

    return df[["user_id", "product_id", "quantity", "weight_gram", "days_between"]]
