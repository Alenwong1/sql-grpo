"""Read-only, time-limited SQLite execution and BIRD-style result comparison.

Safety: the database is opened with `mode=ro`, only a single SELECT / WITH statement is accepted,
and a progress handler aborts any query that runs longer than `timeout_s`.
"""
from __future__ import annotations

import functools
import re
import sqlite3
import time
from dataclasses import dataclass

_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b", re.I)


@dataclass
class ExecResult:
    ok: bool
    rows: list | None = None
    error: str | None = None
    seconds: float = 0.0


def is_safe_select(sql: str) -> bool:
    s = sql.strip().rstrip(";").strip()
    if not s or ";" in s:
        return False  # empty or multiple statements
    if not re.match(r"^\s*(select|with)\b", s, re.I):
        return False
    return not _FORBIDDEN.search(s)


def execute(db_path: str, sql: str, timeout_s: float = 5.0, max_rows: int = 10_000) -> ExecResult:
    if not is_safe_select(sql):
        return ExecResult(False, error="rejected: only a single read-only SELECT is allowed")
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql.strip().rstrip(";"))
        rows = cur.fetchmany(max_rows)
        return ExecResult(True, rows=rows, seconds=time.perf_counter() - t0)
    except sqlite3.OperationalError as e:
        msg = "timeout" if "interrupted" in str(e) else str(e)
        return ExecResult(False, error=msg, seconds=time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001 - any failure is just a failed query for scoring
        return ExecResult(False, error=f"{type(e).__name__}: {e}", seconds=time.perf_counter() - t0)
    finally:
        con.close()


@functools.lru_cache(maxsize=20_000)
def gold_result(db_path: str, sql: str, timeout_s: float = 30.0) -> ExecResult:
    return execute(db_path, sql, timeout_s=timeout_s, max_rows=1_000_000)


def results_match(pred: ExecResult, gold: ExecResult) -> bool:
    """BIRD's execution-accuracy rule: same set of result rows (order and duplicates ignored)."""
    return pred.ok and gold.ok and set(pred.rows) == set(gold.rows)
