"""Execution-match reward for GRPO.

reward = 1.0                 if the predicted query returns the same row set as the gold query
       = valid_bonus (0.1)   if it executes without error but returns a different result
       = 0.0                 if it cannot be parsed out, is rejected, errors or times out

A small valid-SQL bonus gives the policy signal early in training on hard databases; set it to 0
to test whether it lets the model learn "valid but wrong" queries (it is one of the ablations).
"""
from __future__ import annotations

from .prompt import extract_sql
from .sandbox import execute, gold_result, results_match


def score_one(completion, db_path: str, gold_sql: str, valid_bonus: float = 0.1, timeout_s: float = 5.0) -> float:
    sql = extract_sql(completion)
    if not sql:
        return 0.0
    pred = execute(db_path, sql, timeout_s=timeout_s)
    if not pred.ok:
        return 0.0
    return 1.0 if results_match(pred, gold_result(db_path, gold_sql)) else valid_bonus


def make_reward_fn(valid_bonus: float = 0.1, timeout_s: float = 5.0):
    def execution_reward(prompts, completions, db_path, gold_sql, **_) -> list[float]:
        return [score_one(c, d, g, valid_bonus, timeout_s) for c, d, g in zip(completions, db_path, gold_sql)]
    execution_reward.__name__ = "execution_reward"
    return execution_reward
