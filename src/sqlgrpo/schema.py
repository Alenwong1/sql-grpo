"""Schema extraction and retrieval.

Real warehouses have far more tables than fit in a prompt, so the prompt only carries the tables a
BM25 retriever ranks highest for the question (plus tables joined to them by foreign keys).
Column descriptions and a few example values come from BIRD's database_description CSVs.
"""
from __future__ import annotations

import csv
import functools
import os
import re
import sqlite3
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # split camelCase
    return _TOKEN.findall(text.lower().replace("_", " "))


@dataclass
class Column:
    name: str
    type: str
    description: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass
class Table:
    name: str
    columns: list[Column]
    primary_keys: list[str]
    foreign_keys: list[tuple[str, str, str]]  # (column, ref_table, ref_column)


def _descriptions(db_dir: str, table: str) -> dict[str, str]:
    path = os.path.join(db_dir, "database_description", f"{table}.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                for row in csv.DictReader(f):
                    name = (row.get("original_column_name") or "").strip()
                    desc = " ".join(x.strip() for x in (row.get("column_description"), row.get("value_description")) if x and x.strip())
                    if name:
                        out[name.lower()] = re.sub(r"\s+", " ", desc)[:200]
            return out
        except UnicodeDecodeError:
            continue
    return out


@functools.lru_cache(maxsize=256)
def load_schema(db_path: str, n_examples: int = 3) -> tuple[Table, ...]:
    db_dir = os.path.dirname(db_path)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    tables = []
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for t in names:
        info = con.execute(f'PRAGMA table_info("{t}")').fetchall()
        fks = [(r[3], r[2], r[4]) for r in con.execute(f'PRAGMA foreign_key_list("{t}")').fetchall()]
        desc = _descriptions(db_dir, t)
        cols = []
        for _, cname, ctype, _, _, _ in info:
            ex = []
            try:
                ex = [str(r[0])[:40] for r in con.execute(
                    f'SELECT DISTINCT "{cname}" FROM "{t}" WHERE "{cname}" IS NOT NULL LIMIT {n_examples}')]
            except sqlite3.Error:
                pass
            cols.append(Column(cname, ctype or "", desc.get(cname.lower(), ""), ex))
        tables.append(Table(t, cols, [r[1] for r in info if r[5]], fks))
    con.close()
    return tuple(tables)


def render(tables: list[Table], with_examples: bool = True, desc_chars: int = 200, n_examples: int = 3) -> str:
    parts = []
    for t in tables:
        lines = []
        for c in t.columns:
            comment = []
            if c.description and desc_chars:
                comment.append(c.description[:desc_chars])
            if with_examples and c.examples and n_examples:
                comment.append("e.g. " + ", ".join(c.examples[:n_examples]))
            lines.append(f'  "{c.name}" {c.type}' + (f"  -- {'; '.join(comment)}" if comment else ""))
        if t.primary_keys:
            lines.append(f"  PRIMARY KEY ({', '.join(t.primary_keys)})")
        for col, rt, rc in t.foreign_keys:
            lines.append(f'  FOREIGN KEY ("{col}") REFERENCES "{rt}"("{rc}")')
        parts.append(f'CREATE TABLE "{t.name}" (\n' + ",\n".join(lines) + "\n);")
    return "\n\n".join(parts)


def render_budget(tables: list[Table], max_chars: int = 7000) -> str:
    """Render with as much column documentation as fits the prompt budget: full notes, then short
    notes with one example value, then one example only, then bare DDL."""
    for kw in (dict(), dict(desc_chars=60, n_examples=1), dict(desc_chars=0, n_examples=1), dict(desc_chars=0, n_examples=0)):
        text = render(tables, **kw)
        if len(text) <= max_chars:
            return text
    return text


class SchemaRetriever:
    """Rank tables for a question with BM25 over table/column names, descriptions and example values."""

    def __init__(self, tables: tuple[Table, ...]):
        self.tables = list(tables)
        docs = []
        for t in self.tables:
            words = tokenize(t.name) * 3
            for c in t.columns:
                words += tokenize(c.name) * 2 + tokenize(c.description) + tokenize(" ".join(c.examples))
            docs.append(words or ["empty"])
        self.bm25 = BM25Okapi(docs)

    def select(self, question: str, evidence: str = "", top_k: int = 4) -> list[Table]:
        if len(self.tables) <= top_k:
            return self.tables
        scores = self.bm25.get_scores(tokenize(question + " " + evidence))
        order = sorted(range(len(self.tables)), key=lambda i: -scores[i])
        keep = {self.tables[i].name for i in order[:top_k]}
        # add tables one foreign-key hop away from the kept ones, so joins stay possible
        by_name = {t.name.lower(): t for t in self.tables}
        for t in list(self.tables):
            if t.name in keep:
                for _, rt, _ in t.foreign_keys:
                    if rt.lower() in by_name:
                        keep.add(by_name[rt.lower()].name)
        return [t for t in self.tables if t.name in keep]


def gold_tables(sql: str) -> set[str]:
    """Table names referenced by a SQL query (used to measure retrieval recall)."""
    import sqlglot
    from sqlglot import exp
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return set()
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    return {t.name.lower() for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in ctes}
