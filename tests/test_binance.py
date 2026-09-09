from datetime import UTC, datetime

import httpx
import pytest

from cryptopulse.ingestion.binance import SYMBOL_TO_ASSET, BinanceClient, parse_klines


def test_bnb_is_a_supported_market_symbol():
    assert SYMBOL_TO_ASSET["BNBUSDT"] == "BNB"


def test_binance_client_fetches_and_parses_hourly_candles():
    def handle_request(request):
        assert request.url.path == "/api/v3/klines"
        assert request.url.params["symbol"] == "BTCUSDT"
        assert request.url.params["interval"] == "1h"
        assert request.url.params["limit"] == "168"
        return httpx.Response(
            200,
            json=[
                [
                    1_757_318_400_000,
                    "110000.0",
                    "111000.0",
                    "109500.0",
                    "110500.0",
                    "120.0",
                    1_757_321_999_999,
                    "13200000.0",
                    1500,
                    "60.0",
                    "6600000.0",
                    "0",
                ]
            ],
        )

    transport = httpx.MockTransport(handle_request)
    with httpx.Client(
        base_url="https://data-api.binance.vision",
        transport=transport,
    ) as http_client:
        candles = BinanceClient(http_client).fetch_hourly_candles("BTCUSDT")

    assert len(candles) == 1
    assert candles[0].asset == "BTC"
    assert candles[0].candle_timestamp == datetime(2025, 9, 8, 8, tzinfo=UTC)
    assert candles[0].close == 110_500.0
    assert candles[0].trade_count == 1_500


def test_parse_klines_rejects_malformed_rows():
    with pytest.raises(ValueError, match="Invalid Binance kline"):
        parse_klines([[1, "2"]], "BTCUSDT")


def test_binance_client_raises_for_http_errors():
    def handle_request(request):
        return httpx.Response(429, json={"code": -1003}, request=request)

    transport = httpx.MockTransport(handle_request)
    with (
        httpx.Client(base_url="https://example.com", transport=transport) as http_client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        BinanceClient(http_client).fetch_hourly_candles("ETHUSDT")
