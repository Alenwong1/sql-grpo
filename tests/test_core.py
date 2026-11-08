import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sqlgrpo.evaluate import pass_at_k  # noqa: E402
from sqlgrpo.prompt import extract_sql  # noqa: E402
from sqlgrpo.reward import make_reward_fn, score_one  # noqa: E402
from sqlgrpo.sandbox import execute, is_safe_select  # noqa: E402
from sqlgrpo.schema import SchemaRetriever, gold_tables, load_schema, render  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "shop.sqlite"
    con = sqlite3.connect(p)
    con.executescript("""
        CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), amount REAL);
        CREATE TABLE weather(day TEXT, temp REAL);
        INSERT INTO customers VALUES (1,'Ana','LA'),(2,'Bo','SF'),(3,'Cy','LA');
        INSERT INTO orders VALUES (1,1,10.0),(2,1,5.0),(3,2,7.5);
        INSERT INTO weather VALUES ('mon', 20.0);
    """)
    con.commit()
    con.close()
    return str(p)


def test_sandbox_blocks_writes_and_multi_statements(db):
    assert not is_safe_select("DROP TABLE orders")
    assert not is_safe_select("SELECT 1; DELETE FROM orders")
    assert not execute(db, "DELETE FROM orders").ok
    assert execute(db, "SELECT count(*) FROM orders").rows == [(3,)]


def test_sandbox_times_out(db):
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT max(x) FROM c"
    r = execute(db, slow, timeout_s=0.2)
    assert not r.ok and r.error == "timeout"


def test_extract_sql_variants():
    assert extract_sql("Sure:\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("```sql\nSELECT 1\n```\nbetter:\n```sql\nSELECT 2\n```") == "SELECT 2"
    assert extract_sql("The answer is SELECT name FROM t") == "SELECT name FROM t"
    assert extract_sql("no query here") is None
    assert extract_sql([{"role": "assistant", "content": "```sql\nSELECT 3\n```"}]) == "SELECT 3"


def test_reward_levels(db):
    gold = "SELECT name FROM customers WHERE city = 'LA'"
    assert score_one("```sql\nSELECT name FROM customers WHERE city='LA' ORDER BY id DESC\n```", db, gold) == 1.0
    assert score_one("```sql\nSELECT name FROM customers\n```", db, gold) == 0.1
    assert score_one("```sql\nSELECT nope FROM customers\n```", db, gold) == 0.0
    assert score_one("I don't know", db, gold) == 0.0
    fn = make_reward_fn(valid_bonus=0.0)
    assert fn(None, ["```sql\nSELECT name FROM customers\n```"], db_path=[db], gold_sql=[gold]) == [0.0]


def test_schema_render_and_retrieval(db):
    tables = load_schema(db)
    text = render(list(tables))
    assert 'CREATE TABLE "orders"' in text and 'REFERENCES "customers"' in text
    sel = {t.name for t in SchemaRetriever(tables).select("total order amount per customer", top_k=1)}
    assert "orders" in sel and "customers" in sel  # FK neighbour pulled in
    assert "weather" not in sel


def test_gold_tables_ignores_ctes():
    assert gold_tables("WITH x AS (SELECT * FROM a) SELECT * FROM x JOIN b ON 1") == {"a", "b"}


def test_pass_at_k_estimator():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 4) == 1.0
    assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-9
    assert abs(pass_at_k(4, 1, 2) - 0.5) < 1e-9
