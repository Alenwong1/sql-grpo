"""Prompt construction and SQL extraction."""
from __future__ import annotations

import re

SYSTEM = ("You are an expert SQLite analyst. Given a database schema and a question, write one SQLite query "
          "that answers it. Use only tables and columns from the schema. Quote identifiers that contain spaces "
          "with backticks or double quotes.")


def build_messages(schema_text: str, question: str, evidence: str = "") -> list[dict]:
    user = f"Database schema:\n{schema_text}\n\n"
    if evidence:
        user += f"Hint: {evidence}\n\n"
    user += f"Question: {question}\n\nAnswer with a single SQLite query inside a ```sql code block."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def target_message(sql: str) -> list[dict]:
    return [{"role": "assistant", "content": f"```sql\n{sql.strip()}\n```"}]


_BLOCK = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str | None:
    """Last ```sql block if present; otherwise the text from the first SELECT/WITH onward."""
    if isinstance(text, list):  # conversational completion from TRL
        text = text[-1]["content"] if text else ""
    blocks = _BLOCK.findall(text)
    if blocks:
        sql = blocks[-1].strip()
    else:
        m = re.search(r"\b(select|with)\b.*", text, re.I | re.S)
        if not m:
            return None
        sql = m.group(0).strip()
    sql = sql.split(";")[0].strip()
    return sql or None
