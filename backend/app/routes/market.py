# backend/app/routes/market.py

from fastapi import APIRouter, HTTPException, Query
from typing import Literal

from app.services.market_data import get_history, get_quote

router = APIRouter(
    prefix="/api/market",
    tags=["Market"]
)

Exchange = Literal["US", "NSE", "BSE"]


@router.get("/quote/{symbol}")
def quote(symbol: str, exchange: Exchange = Query("US")):
    try:
        return get_quote(symbol, exchange)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/history/{symbol}")
def history(symbol: str, exchange: Exchange = Query("US"), period: str = Query("1mo")):
    try:
        return get_history(symbol, exchange, period)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))