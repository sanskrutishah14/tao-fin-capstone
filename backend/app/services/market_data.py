from typing import Dict, Literal

import yfinance as yf

Exchange = Literal["US", "NSE", "BSE"]

_SUFFIX = {
    "US": "",
    "NSE": ".NS",
    "BSE": ".BO",
}


def _resolve_symbol(symbol: str, exchange: Exchange) -> str:
    symbol = symbol.upper().strip()
    suffix = _SUFFIX.get(exchange, "")

    if suffix and not symbol.endswith(suffix):
        symbol = f"{symbol}{suffix}"

    return symbol


def get_quote(symbol: str, exchange: Exchange = "US") -> Dict:
    resolved = _resolve_symbol(symbol, exchange)

    ticker = yf.Ticker(resolved)

    try:
        fast_info = ticker.fast_info
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch quote for '{resolved}': {e}"
        ) from e

    last_price = fast_info.get("last_price") or fast_info.get("lastPrice")
    previous_close = fast_info.get("previous_close") or fast_info.get("previousClose")

    change = None
    change_pct = None

    if last_price is not None and previous_close:
        change = round(last_price - previous_close, 4)
        change_pct = round((change / previous_close) * 100, 4)

    return {
        "symbol": resolved,
        "exchange": exchange,
        "last_price": last_price,
        "previous_close": previous_close,
        "change": change,
        "change_pct": change_pct,
        "day_high": fast_info.get("day_high"),
        "day_low": fast_info.get("day_low"),
        "volume": fast_info.get("last_volume") or fast_info.get("volume"),
        "market_cap": fast_info.get("market_cap"),
        "currency": fast_info.get("currency"),
    }

def get_financial_ratios(symbol: str, exchange: Exchange = "US") -> Dict:
    """
    Fallback source for named margin ratios when the indexed SEC
    filings don't contain the underlying line items. yfinance's
    .info exposes these as trailing-twelve-month (TTM) figures, which
    is NOT the same thing as a specific fiscal year's audited figure
    -- callers must label this distinction clearly to the user.
    """

    resolved = _resolve_symbol(symbol, exchange)

    try:
        info = yf.Ticker(resolved).info
    except Exception as e:
        raise RuntimeError(f"Could not fetch financial ratios for '{resolved}': {e}") from e

    def pct(key: str):
        v = info.get(key)
        return round(v * 100, 2) if v is not None else None

    return {
        "symbol": resolved,
        "operating_margin_pct": pct("operatingMargins"),
        "gross_margin_pct": pct("grossMargins"),
        "profit_margin_pct": pct("profitMargins"),
        "period": "trailing_twelve_months",
        "note": (
            "Trailing-twelve-month figures from Yahoo Finance, not tied "
            "to a specific fiscal year filing."
        ),
    }


def get_history(symbol: str, exchange: Exchange = "US", period: str = "1mo") -> Dict:
    """period examples: '1d','5d','1mo','3mo','6mo','1y','ytd','max'"""

    resolved = _resolve_symbol(symbol, exchange)
    ticker = yf.Ticker(resolved)

    hist = ticker.history(period=period)

    if hist.empty:
        raise RuntimeError(f"No historical data found for '{resolved}'")

    return {
        "symbol": resolved,
        "exchange": exchange,
        "period": period,
        "points": [
            {
                "date": str(index.date()),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            }
            for index, row in hist.iterrows()
        ],
    }