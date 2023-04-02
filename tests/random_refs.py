import pytest
import requests

from KZ_project.Infrastructure import config
#from ..random_refs import random_sku, random_batchref, random_orderid
import uuid


def random_suffix():
    return uuid.uuid4().hex[:6]


def random_symbol(name=""):
    return f"symbol-{name}-{random_suffix()}"


def random_assetref(name=""):
    return f"asset-{name}-{random_suffix()}"


def random_tracker(name=""):
    return f"tracker-{name}-{random_suffix()}"


def post_to_add_asset(symbol, source):
    url = config.get_api_url()
    r = requests.post(
        f"{url}/add_asset", json={"symbol": symbol, "source": source}
    )
    assert r.status_code == 201