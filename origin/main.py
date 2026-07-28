"""FastAPI 엔트리.  [원본·추정]

실행:
    uvicorn origin.main:app --reload --port 8000

원본에는 lifespan 훅이 없다. (app/api/main.py 에서 MCP 연결을 서버 수명에
맞춰 붙였다 떼려고 추가됐다)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from origin.api.routes import router

app = FastAPI(title="AMHS LangGraph API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/llm/api")
