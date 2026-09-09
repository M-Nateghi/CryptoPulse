import math
from datetime import UTC, datetime

import httpx

from cryptopulse.db.models import MarketCandle

SYMBOL_TO_ASSET = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "BNBUSDT": "BNB",
}


def parse_klines(payload: object, symbol: str) -> list[MarketCandle]:
    """Convert a Binance kline response into market-candle records."""
    normalized_symbol = symbol.upper()
    if normalized_symbol not in SYMBOL_TO_ASSET:
        raise ValueError(f"Unsupported Binance symbol: {symbol}")
    if not isinstance(payload, list):
        raise TypeError("Binance kline response must be a list")

    candles = []
    for index, row in enumerate(payload):
        if not isinstance(row, list):
            raise TypeError(f"Binance kline at index {index} must be a list")
        if len(row) < 9:
            raise ValueError(f"Invalid Binance kline at index {index}")

        try:
            candle = MarketCandle(
                asset=SYMBOL_TO_ASSET[normalized_symbol],
                symbol=normalized_symbol,
                candle_timestamp=datetime.fromtimestamp(row[0] / 1000, tz=UTC),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                quote_volume=float(row[7]),
                trade_count=int(row[8]),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Invalid Binance kline values at index {index}") from error

        numeric_values = (
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.quote_volume,
        )
        if not all(math.isfinite(value) and value >= 0 for value in numeric_values):
            raise ValueError(f"Invalid Binance numeric values at index {index}")
        if candle.high < max(candle.open, candle.close, candle.low):
            raise ValueError(f"Invalid Binance high price at index {index}")
        if candle.low > min(candle.open, candle.close, candle.high):
            raise ValueError(f"Invalid Binance low price at index {index}")
        if candle.trade_count < 0:
            raise ValueError(f"Invalid Binance trade count at index {index}")

        candles.append(candle)

    return candles


class BinanceClient:
    def __init__(self, http_client: httpx.Client) -> None:
        self._http_client = http_client

    def fetch_hourly_candles(
        self,
        symbol: str,
        limit: int = 168,
    ) -> list[MarketCandle]:
        """Fetch recent hourly candles for one supported symbol."""
        if not 1 <= limit <= 1_000:
            raise ValueError("Binance candle limit must be between 1 and 1000")

        response = self._http_client.get(
            "/api/v3/klines",
            params={"symbol": symbol.upper(), "interval": "1h", "limit": limit},
        )
        response.raise_for_status()
        return parse_klines(response.json(), symbol)
