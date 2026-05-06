from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import recommendations, train, segments
from tasks.scheduler import start_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi động scheduler khi app start
    scheduler = start_scheduler()
    yield
    # Tắt scheduler khi app shutdown
    scheduler.shutdown()


app = FastAPI(
    title="Pet Shop AI Service",
    description="AI Service cho website bán phụ kiện thú cưng",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cấu hình lại khi deploy production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendations.router)
app.include_router(train.router)
app.include_router(segments.router)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ai_service"}
