from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from lrts.db import init_db
from lrts.api.prompts import router as prompts_router
from lrts.api.datasets import router as datasets_router
from lrts.api.runs import router as runs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="LRTS",
    description="LLM Regression Testing System — CI/CD for LLM behavior",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(prompts_router)
app.include_router(datasets_router)
app.include_router(runs_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
