"""
Script sinh dữ liệu giả lập cho AI training.
Chạy sau khi đã import seed SQL gốc (08_seed.sql).

Usage:
    python scripts/generate_data.py

Sinh thêm:
    - 195 users (tổng 200)
    - 88 sản phẩm (tổng 100)
    - ~1500 đơn hàng (12 tháng gần đây)
    - ~8000 behavior logs
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from datetime import datetime, timedelta, date
from database import SessionLocal
from sqlalchemy import text

random.seed(42)

# ── Hằng số ──────────────────────────────────────────────────────────────────
N_USERS       = 195   # thêm vào (đã có 5 từ seed)
N_PRODUCTS    = 88    # thêm vào (đã có 12 từ seed)
N_ORDERS      = 1500
N_BEHAVIORS   = 8000
START_DATE    = datetime.now() - timedelta(days=365)

PROVINCES = [
    "TP. Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Cần Thơ",
    "Bình Dương", "Đồng Nai", "Hải Phòng", "Nha Trang",
]
FIRST_NAMES = ["An","Bình","Châu","Dung","Em","Phong","Giang","Hoa",
               "Hùng","Lan","Mai","Nam","Nga","Phúc","Quân","Sơn",
               "Tâm","Thảo","Tuấn","Uyên","Việt","Xuân","Yến","Linh"]
LAST_NAMES  = ["Nguyễn","Trần","Lê","Phạm","Hoàng","Huỳnh","Phan",
               "Vũ","Đặng","Bùi","Đỗ","Hồ","Ngô","Dương","Lý"]

# Sản phẩm mẫu theo danh mục (category_id từ seed)
PRODUCT_TEMPLATES = [
    # (category_id, name_prefix, brand, price_range, weight_gram, is_consumable, pet_types)
    (6,  "Thức ăn hạt chó",        "Royal Canin",  (200000, 900000),  (500,10000),  1, [1]),
    (6,  "Pate chó",               "Pedigree",     (30000,  120000),  (100, 400),   1, [1]),
    (6,  "Snack thưởng chó",       "Cesar",        (40000,  150000),  (50,  300),   1, [1]),
    (7,  "Thức ăn hạt mèo",        "Royal Canin",  (150000, 500000),  (400, 4000),  1, [2]),
    (7,  "Pate mèo",               "Whiskas",      (20000,  100000),  (80,  400),   1, [2]),
    (7,  "Cát vệ sinh mèo",        "Biokat",       (80000,  250000),  (3000,10000), 1, [2]),
    (7,  "Snack thưởng mèo",       "Temptations",  (30000,  90000),   (50,  200),   1, [2]),
    (8,  "Thức ăn cá",             "Tetra",        (30000,  150000),  (50,  500),   1, [3]),
    (2,  "Đồ chơi cắn gặm",        "PetToy",       (30000,  120000),  (50,  300),   0, [1,2]),
    (2,  "Bóng đồ chơi",           "PetToy",       (20000,  80000),   (30,  150),   0, [1,2]),
    (2,  "Đồ chơi treo lồng",      "BirdFun",      (25000,  70000),   (20,  100),   0, [4]),
    (9,  "Vòng cổ thú cưng",       "PetStyle",     (50000,  250000),  (30,  150),   0, [1,2]),
    (9,  "Dây dắt chó",            "PetStyle",     (60000,  200000),  (80,  300),   0, [1]),
    (10, "Áo thú cưng",            "PetFashion",   (80000,  350000),  (50,  200),   0, [1,2]),
    (4,  "Dầu gội thú cưng",       "PetCare",      (60000,  200000),  (200, 500),   1, [1,2]),
    (4,  "Lược chải lông",         "PetCare",      (30000,  120000),  (50,  200),   0, [1,2]),
    (4,  "Bàn chải đánh răng",     "PetDent",      (25000,  80000),   (30,  100),   1, [1,2]),
    (5,  "Ổ nằm thú cưng",         "PetHome",      (150000, 600000),  (300,1500),   0, [1,2]),
    (5,  "Lồng vận chuyển",        "PetHome",      (200000, 800000),  (500,2000),   0, [1,2]),
]

PAYMENT_METHODS = ["cod", "bank_transfer", "momo", "vnpay"]
ORDER_STATUSES  = ["delivered", "delivered", "delivered", "shipping", "cancelled"]
ACTIONS         = ["view", "view", "view", "search", "add_to_cart", "wishlist", "purchase"]


def rand_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def insert_users(db, n: int) -> list[int]:
    """Thêm n users, trả về danh sách user_id mới."""
    print(f"  Inserting {n} users...")
    rows = []
    for i in range(n):
        fn = random.choice(FIRST_NAMES)
        ln = random.choice(LAST_NAMES)
        email = f"user{i+6}_{fn.lower()}@example.com"
        rows.append({
            "full_name": f"{ln} {fn}",
            "email": email,
            "password_hash": "$2b$12$fakehashforseeddataonly000000000000000000000",
            "phone": f"09{random.randint(10000000,99999999)}",
            "role": "customer",
            "is_active": 1,
        })

    db.execute(text("""
        INSERT INTO tbl_users (full_name, email, password_hash, phone, role, is_active)
        VALUES (:full_name, :email, :password_hash, :phone, :role, :is_active)
    """), rows)
    db.commit()

    result = db.execute(text("SELECT pk_user_id FROM tbl_users WHERE role='customer' ORDER BY pk_user_id")).fetchall()
    return [r.pk_user_id for r in result]


def insert_products(db, n: int) -> list[int]:
    """Thêm n sản phẩm, trả về danh sách product_id mới."""
    print(f"  Inserting {n} products...")
    rows = []
    for i in range(n):
        tpl = random.choice(PRODUCT_TEMPLATES)
        cat_id, name_prefix, brand, price_range, weight_range, is_consumable, _ = tpl
        price = random.randint(price_range[0] // 1000, price_range[1] // 1000) * 1000
        sale  = price - random.randint(1, 5) * 10000 if random.random() < 0.4 else None
        weight = random.randint(weight_range[0], weight_range[1])
        slug  = f"{name_prefix.lower().replace(' ','-')}-{i+13}-gen"
        rows.append({
            "fk_category_id": cat_id,
            "name": f"{name_prefix} #{i+13}",
            "slug": slug,
            "description": f"Sản phẩm {name_prefix} chất lượng cao",
            "price": price,
            "sale_price": sale,
            "stock": random.randint(10, 200),
            "sku": f"GEN-{i+13:04d}",
            "brand": brand,
            "weight_gram": weight,
            "is_consumable": is_consumable,
            "is_active": 1,
        })

    db.execute(text("""
        INSERT INTO tbl_products
            (fk_category_id, name, slug, description, price, sale_price,
             stock, sku, brand, weight_gram, is_consumable, is_active)
        VALUES
            (:fk_category_id, :name, :slug, :description, :price, :sale_price,
             :stock, :sku, :brand, :weight_gram, :is_consumable, :is_active)
    """), rows)
    db.commit()

    # Gán pet_type cho sản phẩm mới
    new_products = db.execute(text(
        "SELECT pk_product_id, fk_category_id FROM tbl_products WHERE sku LIKE 'GEN-%'"
    )).fetchall()

    cat_to_pet = {6:[1], 7:[2], 8:[3], 2:[1,2], 9:[1,2], 10:[1,2], 4:[1,2], 5:[1,2]}
    pt_rows = []
    for p in new_products:
        pet_types = cat_to_pet.get(p.fk_category_id, [1])
        for pt in pet_types:
            pt_rows.append({"product_id": p.pk_product_id, "pet_type_id": pt})
    if pt_rows:
        db.execute(text("""
            INSERT IGNORE INTO tbl_product_pet_types (fk_product_id, fk_pet_type_id)
            VALUES (:product_id, :pet_type_id)
        """), pt_rows)
    db.commit()

    result = db.execute(text("SELECT pk_product_id FROM tbl_products ORDER BY pk_product_id")).fetchall()
    return [r.pk_product_id for r in result]


def insert_orders(db, user_ids: list, product_ids: list, n: int):
    """Sinh đơn hàng với pattern thực tế."""
    print(f"  Inserting {n} orders...")

    # Mỗi user có sở thích sản phẩm riêng (simulate preference)
    user_prefs = {uid: random.sample(product_ids, min(10, len(product_ids))) for uid in user_ids}

    for batch_start in range(0, n, 100):
        batch_size = min(100, n - batch_start)
        order_rows = []
        for _ in range(batch_size):
            uid = random.choice(user_ids)
            order_date = rand_date(START_DATE, datetime.now())
            status = random.choice(ORDER_STATUSES)
            pay_method = random.choice(PAYMENT_METHODS)
            pay_status = "paid" if pay_method != "cod" and status == "delivered" else "pending"

            # Chọn 1-4 sản phẩm, ưu tiên sản phẩm user thích
            prefs = user_prefs[uid]
            n_items = random.randint(1, 4)
            chosen = random.sample(prefs, min(n_items, len(prefs)))

            subtotal = sum(random.randint(50000, 500000) for _ in chosen)
            shipping = 25000 if subtotal < 300000 else 0
            total = subtotal + shipping

            order_rows.append({
                "fk_user_id": uid,
                "receiver": f"Người nhận {uid}",
                "phone": f"09{random.randint(10000000,99999999)}",
                "shipping_address": f"Địa chỉ {uid}, {random.choice(PROVINCES)}",
                "subtotal": subtotal,
                "discount_amount": 0,
                "shipping_fee": shipping,
                "total": total,
                "payment_method": pay_method,
                "payment_status": pay_status,
                "order_status": status,
                "created_at": order_date,
                "updated_at": order_date,
            })

        db.execute(text("""
            INSERT INTO tbl_orders
                (fk_user_id, receiver, phone, shipping_address, subtotal,
                 discount_amount, shipping_fee, total, payment_method,
                 payment_status, order_status, created_at, updated_at)
            VALUES
                (:fk_user_id, :receiver, :phone, :shipping_address, :subtotal,
                 :discount_amount, :shipping_fee, :total, :payment_method,
                 :payment_status, :order_status, :created_at, :updated_at)
        """), order_rows)
        db.commit()

    # Thêm order_items
    print("  Inserting order items...")
    orders = db.execute(text(
        "SELECT pk_order_id, fk_user_id FROM tbl_orders WHERE pk_order_id > 5"
    )).fetchall()

    item_rows = []
    for order in orders:
        prefs = user_prefs.get(order.fk_user_id, product_ids[:10])
        n_items = random.randint(1, 3)
        chosen = random.sample(prefs, min(n_items, len(prefs)))
        for pid in chosen:
            price = random.randint(50000, 500000)
            item_rows.append({
                "fk_order_id": order.pk_order_id,
                "fk_product_id": pid,
                "product_name": f"Sản phẩm {pid}",
                "unit_price": price,
                "quantity": random.randint(1, 3),
            })
        if len(item_rows) >= 500:
            db.execute(text("""
                INSERT INTO tbl_order_items
                    (fk_order_id, fk_product_id, product_name, unit_price, quantity)
                VALUES
                    (:fk_order_id, :fk_product_id, :product_name, :unit_price, :quantity)
            """), item_rows)
            db.commit()
            item_rows = []

    if item_rows:
        db.execute(text("""
            INSERT INTO tbl_order_items
                (fk_order_id, fk_product_id, product_name, unit_price, quantity)
            VALUES
                (:fk_order_id, :fk_product_id, :product_name, :unit_price, :quantity)
        """), item_rows)
        db.commit()


def insert_behaviors(db, user_ids: list, product_ids: list, n: int):
    """
    Sinh behavior logs với pattern có ý nghĩa:
    - User nuôi chó (id lẻ) xem nhiều sản phẩm chó
    - User nuôi mèo (id chẵn) xem nhiều sản phẩm mèo
    - Sản phẩm tiêu hao được xem/mua lặp lại
    """
    print(f"  Inserting {n} behavior logs...")

    # Lấy sản phẩm theo loại
    dog_products = db.execute(text("""
        SELECT DISTINCT fk_product_id FROM tbl_product_pet_types WHERE fk_pet_type_id = 1
    """)).fetchall()
    cat_products = db.execute(text("""
        SELECT DISTINCT fk_product_id FROM tbl_product_pet_types WHERE fk_pet_type_id = 2
    """)).fetchall()
    dog_pids = [r.fk_product_id for r in dog_products]
    cat_pids = [r.fk_product_id for r in cat_products]

    rows = []
    for _ in range(n):
        uid = random.choice(user_ids)
        session_id = f"sess-{uid}-{random.randint(1000,9999)}"

        # User lẻ -> thích chó, user chẵn -> thích mèo
        if uid % 2 == 1 and dog_pids:
            pool = dog_pids if random.random() < 0.7 else product_ids
        elif cat_pids:
            pool = cat_pids if random.random() < 0.7 else product_ids
        else:
            pool = product_ids

        pid = random.choice(pool)
        action = random.choice(ACTIONS)
        log_date = rand_date(START_DATE, datetime.now())

        rows.append({
            "fk_user_id": uid,
            "session_id": session_id,
            "fk_product_id": pid if action != "search" else None,
            "action": action,
            "search_query": f"từ khoá {random.randint(1,20)}" if action == "search" else None,
            "duration_sec": random.randint(5, 180) if action == "view" else None,
            "created_at": log_date,
        })

        if len(rows) >= 500:
            db.execute(text("""
                INSERT INTO tbl_user_behavior_logs
                    (fk_user_id, session_id, fk_product_id, action, search_query, duration_sec, created_at)
                VALUES
                    (:fk_user_id, :session_id, :fk_product_id, :action, :search_query, :duration_sec, :created_at)
            """), rows)
            db.commit()
            rows = []

    if rows:
        db.execute(text("""
            INSERT INTO tbl_user_behavior_logs
                (fk_user_id, session_id, fk_product_id, action, search_query, duration_sec, created_at)
            VALUES
                (:fk_user_id, :session_id, :fk_product_id, :action, :search_query, :duration_sec, :created_at)
        """), rows)
        db.commit()


def main():
    print("=== Bắt đầu sinh dữ liệu giả lập ===\n")
    db = SessionLocal()
    try:
        print("[1/4] Users...")
        user_ids = insert_users(db, N_USERS)
        print(f"      Tổng users: {len(user_ids)}\n")

        print("[2/4] Products...")
        product_ids = insert_products(db, N_PRODUCTS)
        print(f"      Tổng products: {len(product_ids)}\n")

        print("[3/4] Orders & Order Items...")
        insert_orders(db, user_ids, product_ids, N_ORDERS)
        order_count = db.execute(text("SELECT COUNT(*) FROM tbl_orders")).scalar()
        print(f"      Tổng orders: {order_count}\n")

        print("[4/4] Behavior Logs...")
        insert_behaviors(db, user_ids, product_ids, N_BEHAVIORS)
        log_count = db.execute(text("SELECT COUNT(*) FROM tbl_user_behavior_logs")).scalar()
        print(f"      Tổng behavior logs: {log_count}\n")

        print("=== Hoàn thành! ===")
        print("Bây giờ bạn có thể gọi POST /train/all để huấn luyện các mô hình AI.")

    except Exception as e:
        db.rollback()
        print(f"Lỗi: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
