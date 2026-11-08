"""End-to-end smoke tests of SFT and every GRPO variant on a tiny random model (CPU, ~1 min).
They check wiring (data -> prompts -> rewards -> optimizer steps), not learning."""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TINY = os.environ.get("TINY_MODEL", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")


@pytest.fixture(scope="module")
def bird_like(tmp_path_factory):
    """A miniature BIRD-format split: data/<split>/<split>.json + <split>_databases/<db>/<db>.sqlite"""
    import json
    root = tmp_path_factory.mktemp("bird")
    for split in ("train", "dev"):
        db_dir = root / split / f"{split}_databases" / "shop"
        db_dir.mkdir(parents=True)
        con = sqlite3.connect(db_dir / "shop.sqlite")
        con.executescript("CREATE TABLE items(id INTEGER PRIMARY KEY, name TEXT, price REAL);"
                          "INSERT INTO items VALUES (1,'pen',1.5),(2,'book',12.0),(3,'bag',30.0);")
        con.commit()
        con.close()
        qs = [{"question_id": i, "db_id": "shop", "question": q, "evidence": "", "SQL": s, "difficulty": "simple"}
              for i, (q, s) in enumerate([("How many items are there?", "SELECT count(*) FROM items"),
                                          ("What is the most expensive item?", "SELECT name FROM items ORDER BY price DESC LIMIT 1"),
                                          ("List items under 10 dollars.", "SELECT name FROM items WHERE price < 10"),
                                          ("What is the average price?", "SELECT avg(price) FROM items")])]
        (root / split / f"{split}.json").write_text(json.dumps(qs))
    return str(root)


def test_sft_runs(bird_like, tmp_path):
    from sqlgrpo.sft import main
    main(["--model", TINY, "--data", bird_like, "--out", str(tmp_path / "sft"), "--max-steps", "2",
          "--batch", "2", "--grad-accum", "1", "--max-length", "512", "--lora-r", "4"])


@pytest.mark.parametrize("variant", ["grpo_seq", "dr_grpo", "dapo_cliphigher", "lite_ppo", "gspo"])
def test_grpo_variant_runs(bird_like, tmp_path, variant):
    from sqlgrpo.grpo import main
    main(["--model", TINY, "--data", bird_like, "--out", str(tmp_path / variant), "--variant", variant,
          "--max-steps", "1", "--batch", "4", "--grad-accum", "1", "--num-generations", "4",
          "--max-completion-length", "16", "--logging-steps", "1", "--save-steps", "1000"])


def test_evaluate_runs(bird_like, tmp_path):
    from sqlgrpo.evaluate import main
    out = tmp_path / "eval.json"
    main(["--model", TINY, "--data", bird_like, "--backend", "hf", "--n", "2", "--max-new-tokens", "16", "--out", str(out)])
    import json
    r = json.loads(out.read_text())
    assert 0.0 <= r["EX"] <= 1.0 and "pass@k" in r
