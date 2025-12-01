import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from KZ_project.Infrastructure.services.binance_service.binance_client import (
    BinanceClient,
)
from KZ_project.webapi.entrypoints.japanese_app import japanese_router

load_dotenv()

app = FastAPI(title="KZ Forecasting Engine", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("BINANCE_API_KEY")
api_secret_key = os.getenv("BINANCE_SECRET_KEY")
_binance_client = (
    BinanceClient(api_key, api_secret_key) if api_key and api_secret_key else None
)


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


@app.get("/binance/price/{symbol}")
async def get_binance_price(symbol: str):
    if _binance_client is None:
        raise HTTPException(
            status_code=503,
            detail="Binance client not configured. Set BINANCE_API_KEY and BINANCE_SECRET_KEY.",
        )

    return {"symbol": symbol, "price": _binance_client.ticker_price(symbol)}


app.include_router(japanese_router, prefix="/data")
