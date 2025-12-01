import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("binance")

from KZ_project.Infrastructure.services.binance_service.binance_client import BinanceClient


class DummyBinanceClient:
    def __init__(self, price: str = "123.45", klines: list | None = None):
        self.price = price
        self.klines = klines or [
            [
                1622505600000,  # open time
                "100",
                "110",
                "90",
                "105",
                "10",
                1622509200000,  # close time
                "0",
                1,
                "0",
                "0",
                "0",
            ]
        ]
        self.requested_symbols = []

    def get_symbol_ticker(self, symbol: str):
        self.requested_symbols.append(symbol)
        return {"symbol": symbol, "price": self.price}

    def get_historical_klines(self, symbol, interval, start_str, end_str=None, limit=1000):
        self.requested_symbols.append(symbol)
        return list(self.klines)


class DummyWebsocketManager:
    def start(self):
        return None

    def stop(self):
        return None



def test_ticker_price_uses_stubbed_client():
    stub_client = DummyBinanceClient(price="25000.00")
    client = BinanceClient(
        api_key="dummy",
        api_secret_key="dummy",
        client=stub_client,
        twm=DummyWebsocketManager(),
    )

    price = client.ticker_price("BTCUSDT")

    assert price == 25000.00
    assert stub_client.requested_symbols == ["BTCUSDT"]


def test_get_history_formats_dataframe_with_numeric_columns():
    klines = [
        [
            1622505600000,
            "100",
            "110",
            "90",
            "105",
            "10",
            1622509200000,
            "0",
            1,
            "0",
            "0",
            "0",
        ]
    ]
    stub_client = DummyBinanceClient(klines=klines)
    client = BinanceClient(
        api_key="dummy",
        api_secret_key="dummy",
        client=stub_client,
        twm=DummyWebsocketManager(),
    )

    df = client.get_history("BTCUSDT", "1h", "2021-06-01 00:00:00")

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0].isoformat() == "2021-06-01T00:00:00+00:00"
    assert df.loc[df.index[0], "close"] == 105.0
    assert set(df.columns) == {"open", "high", "low", "close", "volume", "adj_close"}
