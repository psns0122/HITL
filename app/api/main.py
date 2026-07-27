"""FastAPI 엔트리.

실행:
    uvicorn app.api.main:app --reload --port 8000

lifespan 으로 MCP 연결을 서버 수명에 맞춰 붙였다 뗀다.
요청마다 붙으면 느리고, 안 떼면 종료 시 세션이 남는다.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app._mcp import mcp_manager
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

app.include_router(router, prefix="/llm/api")
