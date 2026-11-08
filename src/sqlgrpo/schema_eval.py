"""How often does the retrieved schema contain every table the gold query uses?

    python -m sqlgrpo.schema_eval --data data/bird --split dev
"""
import argparse
import json

from .data import db_path, load_examples
from .schema import SchemaRetriever, gold_tables, load_schema


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bird")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="results/schema_recall.json")
    a = ap.parse_args(argv)
    ex = load_examples(a.data, a.split)
    res = {}
    for k in (2, 3, 4, 5, 6):
        hit = n = kept = 0
        for e in ex:
            tables = load_schema(db_path(a.data, a.split, e["db_id"]))
            sel = {t.name.lower() for t in SchemaRetriever(tables).select(e["question"], e.get("evidence", ""), k)}
            g = gold_tables(e["SQL"])
            if g:
                n += 1
                hit += g <= sel
                kept += len(sel)
        res[f"top_{k}"] = {"all_gold_tables_recall": hit / n, "avg_tables_kept": kept / n, "questions": n}
    print(json.dumps(res, indent=2))
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
