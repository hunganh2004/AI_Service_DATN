"""
Association Rules Mining dùng FP-Growth (mlxtend).
Tìm các sản phẩm thường được mua cùng nhau.
Kết quả được lưu vào tbl_association_rules.
"""
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sqlalchemy.orm import Session

from data.preprocessor import load_order_items

MIN_SUPPORT = 0.01       # Xuất hiện trong ít nhất 1% đơn hàng
MIN_CONFIDENCE = 0.2     # Xác suất mua B khi đã mua A >= 20%
MIN_LIFT = 1.0           # Lift > 1 nghĩa là có mối liên hệ thực sự


def train(db: Session) -> dict:
    """
    Chạy FP-Growth, lưu kết quả vào tbl_association_rules.
    Trả về số lượng rules tìm được.
    """
    df = load_order_items()
    if df.empty:
        return {"error": "Không có dữ liệu đơn hàng"}

    # Nhóm sản phẩm theo đơn hàng
    baskets = df.groupby("order_id")["product_id"].apply(list).tolist()

    if len(baskets) < 10:
        return {"error": "Chưa đủ đơn hàng để khai phá luật kết hợp"}

    # Encode thành ma trận nhị phân
    te = TransactionEncoder()
    te_array = te.fit_transform(baskets)
    basket_df = pd.DataFrame(te_array, columns=te.columns_)

    # FP-Growth
    frequent_items = fpgrowth(basket_df, min_support=MIN_SUPPORT, use_colnames=True)
    if frequent_items.empty:
        return {"rules_saved": 0, "message": "Không tìm được itemset phổ biến"}

    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE,
    )
    rules = rules[rules["lift"] >= MIN_LIFT]

    # Chỉ giữ luật 1 antecedent -> 1 consequent (đơn giản, dễ dùng)
    rules = rules[
        rules["antecedents"].apply(len) == 1
    ][
        rules["consequents"].apply(len) == 1
    ].copy()

    rules["antecedent_id"] = rules["antecedents"].apply(lambda x: int(list(x)[0]))
    rules["consequent_id"] = rules["consequents"].apply(lambda x: int(list(x)[0]))

    # Lưu vào DB (xoá cũ, insert mới)
    db.execute(
        __import__("sqlalchemy").text("DELETE FROM tbl_association_rules")
    )

    rows = rules[["antecedent_id", "consequent_id", "support", "confidence", "lift"]].to_dict("records")
    if rows:
        db.execute(
            __import__("sqlalchemy").text("""
                INSERT INTO tbl_association_rules
                    (fk_antecedent, fk_consequent, support, confidence, lift)
                VALUES
                    (:antecedent_id, :consequent_id, :support, :confidence, :lift)
            """),
            rows,
        )
    db.commit()

    return {"rules_saved": len(rows)}


def get_associated_products(product_id: int, db: Session, top_n: int = 5) -> list[dict]:
    """Lấy sản phẩm thường mua kèm với product_id từ DB."""
    from sqlalchemy import text
    result = db.execute(
        text("""
            SELECT fk_consequent AS product_id, confidence, lift
            FROM tbl_association_rules
            WHERE fk_antecedent = :pid
            ORDER BY lift DESC, confidence DESC
            LIMIT :n
        """),
        {"pid": product_id, "n": top_n},
    ).fetchall()

    return [
        {"product_id": row.product_id, "confidence": round(row.confidence, 4), "lift": round(row.lift, 4)}
        for row in result
    ]
