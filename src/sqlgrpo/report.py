"""Collect results/*.json from evaluate.py into one markdown table: python -m sqlgrpo.report results"""
import glob
import json
import os
import sys


def main(d="results"):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        if "EX" not in r:
            continue
        pk = r.get("pass@k", {})
        diff = r.get("EX_by_difficulty", {})
        rows.append(f"| {os.path.basename(f)[:-5]} | {r['EX']:.3f} | {diff.get('simple', float('nan')):.3f} | "
                    f"{diff.get('moderate', float('nan')):.3f} | {diff.get('challenging', float('nan')):.3f} | "
                    f"{pk.get('pass@1', float('nan')):.3f} | {pk.get('pass@16', float('nan')):.3f} | {r['n_questions']} |")
    print("| run | EX (greedy) | simple | moderate | challenging | pass@1 (T=1) | pass@16 | questions |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    print("\n".join(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
