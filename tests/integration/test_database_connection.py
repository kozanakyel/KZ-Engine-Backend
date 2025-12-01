import os

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from KZ_project.Infrastructure.config import get_postgres_uri


@pytest.mark.skipif(not os.getenv("PG_DB_URL"), reason="PG_DB_URL not configured")
def test_can_connect_to_postgres_and_query():
    uri = get_postgres_uri()
    with psycopg2.connect(uri) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            [(value,)] = cur.fetchall()

    assert value == 1
