"""
Endpoints phân cụm khách hàng:
  GET /segments/user/{user_id}  - Lấy segment của user
  GET /segments/all             - Danh sách tất cả segments và số lượng user
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from database import get_db
from models.clustering import predict_segment

router = APIRouter(prefix="/segments", tags=["Segments"])


@router.get("/user/{user_id}")
def get_user_segment(user_id: int, db: Session = Depends(get_db)):
    return predict_segment(user_id, db)


@router.get("/all")
def get_all_segments(db: Session = Depends(get_db)):
    rows = db.execute(
        text("""
            SELECT cs.pk_segment_id, cs.name, cs.description,
                   COUNT(us.fk_user_id) AS user_count
            FROM tbl_customer_segments cs
            LEFT JOIN tbl_user_segments us ON us.fk_segment_id = cs.pk_segment_id
            GROUP BY cs.pk_segment_id
        """)
    ).fetchall()
    return {
        "segments": [
            {"segment_id": r.pk_segment_id, "name": r.name, "description": r.description, "user_count": r.user_count}
            for r in rows
        ]
    }
