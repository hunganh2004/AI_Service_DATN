"""
Endpoints gợi ý sản phẩm:
  GET  /recommendations/homepage      - Gợi ý trang chủ (collaborative + trending fallback)
  GET  /recommendations/product/{id}  - Sản phẩm mua kèm (association rules)
  GET  /recommendations/repurchase    - Nhắc mua lại sản phẩm tiêu hao
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from database import get_db, fetch_df
from models import collaborative, association, repurchase

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


def _get_trending(db: Session, top_n: int) -> list[dict]:
    """Fallback: lấy sản phẩm bán chạy nhất."""
    rows = db.execute(
        text("""
            SELECT fk_product_id AS product_id, SUM(quantity) AS sold
            FROM tbl_order_items oi
            JOIN tbl_orders o ON o.pk_order_id = oi.fk_order_id
            WHERE o.order_status = 'delivered'
            GROUP BY fk_product_id
            ORDER BY sold DESC
            LIMIT :n
        """),
        {"n": top_n},
    ).fetchall()
    return [{"product_id": r.product_id, "score": float(r.sold), "rec_type": "trending"} for r in rows]


@router.get("/homepage")
def homepage_recommendations(
    user_id: int = Query(..., description="ID người dùng"),
    top_n: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """
    Gợi ý sản phẩm cho trang chủ.
    Ưu tiên collaborative filtering, fallback về trending nếu chưa đủ dữ liệu.
    """
    try:
        preds = collaborative.predict_for_user(user_id, top_n)
        # Nếu tất cả score = 0 (cold start hoàn toàn), dùng trending
        if not preds or all(p["score"] == 0 for p in preds):
            preds = _get_trending(db, top_n)
            rec_type = "trending"
        else:
            rec_type = "collaborative"
    except FileNotFoundError:
        preds = _get_trending(db, top_n)
        rec_type = "trending"

    # Lưu vào tbl_product_recommendations
    _save_recommendations(user_id, preds, rec_type, db)

    return {"user_id": user_id, "rec_type": rec_type, "recommendations": preds}


@router.get("/product/{product_id}")
def product_recommendations(
    product_id: int,
    top_n: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """Sản phẩm thường mua kèm với product_id (Association Rules)."""
    results = association.get_associated_products(product_id, db, top_n)
    # Thêm field `score` (= confidence) để tương thích với backend
    for r in results:
        r["score"] = r["confidence"]
    return {"product_id": product_id, "rec_type": "association", "recommendations": results}


@router.get("/repurchase")
def repurchase_reminders(
    user_id: int = Query(...),
    days_ahead: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """Sản phẩm tiêu hao cần mua lại trong days_ahead ngày tới."""
    results = repurchase.get_upcoming_repurchases(user_id, db, days_ahead)
    return {"user_id": user_id, "days_ahead": days_ahead, "reminders": results}


def _save_recommendations(user_id: int, preds: list[dict], rec_type: str, db: Session):
    """Lưu kết quả gợi ý vào DB (xoá cũ theo user+type, insert mới)."""
    db.execute(
        text("DELETE FROM tbl_product_recommendations WHERE fk_user_id = :uid AND rec_type = :rt"),
        {"uid": user_id, "rt": rec_type},
    )
    rows = [
        {"user_id": user_id, "product_id": p["product_id"], "score": p["score"], "rec_type": rec_type}
        for p in preds
    ]
    if rows:
        db.execute(
            text("""
                INSERT INTO tbl_product_recommendations (fk_user_id, fk_product_id, score, rec_type)
                VALUES (:user_id, :product_id, :score, :rec_type)
            """),
            rows,
        )
    db.commit()
