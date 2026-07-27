"""FastAPI 엔트리 — shared_code.md §5 구조 유지."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(title="AMHS LangGraph API (HITL ActionAgent)")

# 로컬 Streamlit 에서 호출하므로 CORS 허용 (사내 반입 시 tighten)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/llm/api")
