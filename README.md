# Pet Shop AI Service

AI Service cho website bán phụ kiện thú cưng, cung cấp các tính năng gợi ý sản phẩm, phân cụm khách hàng và dự đoán tái mua hàng.

## Tính năng

- **Collaborative Filtering** – Gợi ý sản phẩm dựa trên hành vi người dùng (SVD)
- **Association Rules** – Gợi ý sản phẩm thường mua kèm (Apriori/FP-Growth)
- **Customer Clustering** – Phân cụm khách hàng theo hành vi mua sắm (K-Means)
- **Repurchase Prediction** – Dự đoán khả năng mua lại sản phẩm (Random Forest)
- **Auto Retrain** – Tự động huấn luyện lại model theo lịch (APScheduler)

## Yêu cầu

- Python 3.10+
- MySQL 8.0+

## Cài đặt

```bash
# 1. Clone repo
git clone <repo-url>

# 2. Tạo virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Cấu hình biến môi trường
cp .env.example .env
# Chỉnh sửa .env với thông tin database của bạn
```

## Cấu hình

Tạo file `.env` từ `.env.example`:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=pet_accessory_shop
```

## Chạy service

```bash
uvicorn main:app --reload --port 8001
```

API docs tại: `http://localhost:8001/docs`

## Cấu trúc project

```
ai_service/
├── main.py              # Entry point FastAPI
├── config.py            # Cấu hình từ .env
├── database.py          # SQLAlchemy engine & session
├── requirements.txt
├── .env.example
├── routers/             # API endpoints
│   ├── recommendations.py
│   ├── segments.py
│   └── train.py
├── models/              # Logic ML models
│   ├── collaborative.py
│   ├── association.py
│   ├── clustering.py
│   ├── repurchase.py
│   └── saved/           # File .pkl (không commit)
├── data/
│   └── preprocessor.py  # Tiền xử lý dữ liệu
├── tasks/
│   └── scheduler.py     # Auto retrain scheduler
└── scripts/
    └── generate_data.py # Script tạo dữ liệu mẫu
```

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/health` | Kiểm tra trạng thái service |
| GET | `/recommendations/...` | Lấy gợi ý sản phẩm |
| POST | `/train/...` | Huấn luyện lại model |
| GET | `/segments/...` | Thông tin phân cụm khách hàng |
