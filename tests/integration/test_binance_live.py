import os

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("binance")

from KZ_project.Infrastructure.services.binance_service.binance_client import BinanceClient


@pytest.mark.skipif(
    not (os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_SECRET_KEY")),
    reason="Binance credentials not configured",
)
def test_live_ticker_price_positive():
    client = BinanceClient(
        api_key=os.environ["BINANCE_API_KEY"],
        api_secret_key=os.environ["BINANCE_SECRET_KEY"],
    )

    price = client.ticker_price("BTCUSDT")

    assert price > 0
