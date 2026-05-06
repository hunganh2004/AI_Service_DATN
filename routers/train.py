"""
Endpoints kích hoạt huấn luyện mô hình:
  POST /train/collaborative
  POST /train/association
  POST /train/clustering
  POST /train/repurchase
  POST /train/all
"""
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from database import get_db
from models import collaborative, association, clustering, repurchase

router = APIRouter(prefix="/train", tags=["Training"])


@router.post("/collaborative")
def train_collaborative():
    """Huấn luyện SVD Collaborative Filtering."""
    result = collaborative.train()
    return {"model": "collaborative", **result}


@router.post("/association")
def train_association(db: Session = Depends(get_db)):
    """Chạy FP-Growth và lưu Association Rules vào DB."""
    result = association.train(db)
    return {"model": "association", **result}


@router.post("/clustering")
def train_clustering(
    n_clusters: int = None,
    db: Session = Depends(get_db),
):
    """Phân cụm khách hàng bằng K-Means."""
    result = clustering.train(db, n_clusters)
    return {"model": "clustering", **result}


@router.post("/repurchase")
def train_repurchase(db: Session = Depends(get_db)):
    """Huấn luyện Random Forest dự đoán mua lại."""
    result = repurchase.train(db)
    return {"model": "repurchase", **result}


@router.post("/all")
def train_all(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Huấn luyện tất cả mô hình (chạy tuần tự trong background)."""
    def _run_all():
        collaborative.train()
        association.train(db)
        clustering.train(db)
        repurchase.train(db)

    background_tasks.add_task(_run_all)
    return {"message": "Đã bắt đầu huấn luyện tất cả mô hình trong background"}
