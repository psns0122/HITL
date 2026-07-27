"""FastAPI 엔트리.

실행:
    uvicorn app.api.main:app --reload --port 8000

lifespan 으로 MCP 연결을 서버 수명에 맞춰 붙였다 뗀다.
요청마다 붙으면 느리고, 안 떼면 종료 시 세션이 남는다.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app._mcp import mcp_manager
from app.api import limits
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 기동/종료 훅."""
    # --- startup
    await mcp_manager.connect()

    yield

    # --- shutdown
    await mcp_manager.disconnect()


app = FastAPI(
    title="AMHS LangGraph API (HITL ActionAgent)",
    lifespan=lifespan,
)

# 로컬 Streamlit 에서 호출하므로 전부 허용해 둔다.
# 사내 반입 시에는 allow_origins 를 실제 프론트 주소로 좁힐 것.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """클라이언트(IP)당 분당 요청 수를 제한한다.

    스트리밍 엔드포인트에만 적용하고, 넘치면 429 로 즉시 거절한다.
    (동시성 제한은 대기, 유량 제한은 거절 — 성격이 달라 분리했다.)
    """
    if request.url.path.endswith("/chat/stream"):
        client_id = request.client.host if request.client else "unknown"
        ok, retry_after = limits.rate_limit_ok(client_id)
        if not ok:
            return JSONResponse(
                status_code=429,
                content={"error": "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.",
                         "retry_after": retry_after},
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)


app.include_router(router, prefix="/llm/api")
