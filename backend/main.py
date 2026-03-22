"""
main.py – Sentinel-X FastAPI Backend

Endpoints:
    GET  /               Health ping
    GET  /health         Health check (JSON)
    POST /predict/image  Top-1 classification only
    POST /attack/preview Lightweight FGSM preview (no PGD, base64 response)
    POST /attack/image   Full FGSM + PGD pipeline with caching, DB logging, top-5

Environment variables (see .env.example):
    REDIS_URL       – Redis connection URL  (default: redis://localhost:6379)
    POSTGRES_DSN    – PostgreSQL DSN        (default: local sentinelx DB)
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import cache as cache_module
import db as db_module
from image.model import load_model
from image.router import router as image_router

# Load .env before anything else
load_dotenv()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lifespan – startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ── startup ──────────────────────────────────────────────────────────
    print("Loading ResNet-18 weights …")
    load_model()
    print("Model ready.")

    # Redis (non-fatal if unavailable — caching is best-effort)
    try:
        await cache_module.init_redis()
    except Exception as exc:
        print(f"[WARNING] Redis unavailable — caching disabled: {exc}")

    # PostgreSQL (non-fatal — logging is best-effort)
    try:
        await db_module.init_db()
    except Exception as exc:
        print(f"[WARNING] PostgreSQL unavailable — DB logging disabled: {exc}")

    yield

    # ── shutdown ─────────────────────────────────────────────────────────
    await cache_module.close_redis()
    await db_module.close_db()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sentinel-X – Adversarial ML Demo",
    description=(
        "Upload an image to classify it with ResNet-18 and see how "
        "FGSM / PGD adversarial attacks change the prediction."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image_router)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
