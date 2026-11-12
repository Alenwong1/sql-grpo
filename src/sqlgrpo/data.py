"""BIRD loading -> HF Datasets with chat prompts.

Expected layout (scripts/download_bird.sh creates it):
  data/bird/train/train.json  + data/bird/train/train_databases/<db_id>/<db_id>.sqlite
  data/bird/dev/dev.json      + data/bird/dev/dev_databases/<db_id>/<db_id>.sqlite
"""
from __future__ import annotations

import json
import os

from datasets import Dataset

from .prompt import build_messages, target_message
from .schema import SchemaRetriever, load_schema, render_budget

SPLITS = {"train": ("train.json", "train_databases"), "dev": ("dev.json", "dev_databases")}


def db_path(root: str, split: str, db_id: str) -> str:
    return os.path.join(root, split, SPLITS[split][1], db_id, f"{db_id}.sqlite")


def load_examples(root: str, split: str, limit: int | None = None, db_ids: list[str] | None = None) -> list[dict]:
    with open(os.path.join(root, split, SPLITS[split][0])) as f:
        ex = json.load(f)
    if db_ids:
        ex = [e for e in ex if e["db_id"] in set(db_ids)]
    return ex[:limit] if limit else ex


def build_dataset(root: str, split: str, limit: int | None = None, top_k_tables: int = 4,
                  with_target: bool = False, db_ids: list[str] | None = None) -> Dataset:
    rows = []
    retrievers: dict[str, SchemaRetriever] = {}
    for e in load_examples(root, split, limit, db_ids):
        path = db_path(root, split, e["db_id"])
        if e["db_id"] not in retrievers:
            retrievers[e["db_id"]] = SchemaRetriever(load_schema(path))
        tables = retrievers[e["db_id"]].select(e["question"], e.get("evidence", ""), top_k_tables)
        row = {
            "prompt": build_messages(render_budget(tables), e["question"], e.get("evidence", "")),
            "db_path": path, "gold_sql": e["SQL"], "db_id": e["db_id"],
            "difficulty": e.get("difficulty", "unknown"), "question_id": e.get("question_id", len(rows)),
            "kept_tables": [t.name for t in tables],
        }
        if with_target:
            row["completion"] = target_message(e["SQL"])
        rows.append(row)
    return Dataset.from_list(rows)
