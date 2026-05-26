"""
Lên lịch tự động train lại mô hình và gửi thông báo nhắc mua lại.
- Collaborative: mỗi 6 giờ
- Association + Clustering + Repurchase: mỗi ngày lúc 2:00 AM
- Repurchase notifications: mỗi ngày lúc 8:00 AM
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from database import SessionLocal
from models import collaborative, association, clustering, repurchase


def _train_collaborative():
    try:
        result = collaborative.train()
        print(f"[Scheduler] Collaborative: {result}")
    except Exception as e:
        print(f"[Scheduler] Collaborative error: {e}")


def _train_daily():
    db = SessionLocal()
    try:
        r1 = association.train(db)
        r2 = clustering.train(db)
        r3 = repurchase.train(db)
        print(f"[Scheduler] Daily train - Association: {r1}, Clustering: {r2}, Repurchase: {r3}")
    except Exception as e:
        print(f"[Scheduler] Daily train error: {e}")
    finally:
        db.close()


def _send_notifications():
    db = SessionLocal()
    try:
        result = repurchase.send_repurchase_notifications(db)
        print(f"[Scheduler] Repurchase notifications: {result}")
    except Exception as e:
        print(f"[Scheduler] Notification error: {e}")
    finally:
        db.close()


def start_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        _train_collaborative,
        trigger=IntervalTrigger(hours=6),
        id="train_collaborative",
        replace_existing=True,
    )
    scheduler.add_job(
        _train_daily,
        trigger=CronTrigger(hour=2, minute=0),
        id="train_daily",
        replace_existing=True,
    )
    scheduler.add_job(
        _send_notifications,
        trigger=CronTrigger(hour=8, minute=0),
        id="send_repurchase_notifications",
        replace_existing=True,
    )

    scheduler.start()
    return scheduler
