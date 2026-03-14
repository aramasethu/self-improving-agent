"""
Compare two evaluation runs to show improvement/regression.

Usage:
  uv run compare_runs.py                          # compare the two most recent runs
  uv run compare_runs.py eval_baseline.json eval_improved.json  # compare specific runs
"""

import json
import os
import sys

EVAL_DIR = "results/evaluations"


def load_eval(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(before: dict, after: dict, before_name: str, after_name: str):
    print(f"\n{'='*70}")
    print(f"Comparison: {before_name} → {after_name}")
    print(f"{'='*70}")

    # Batch summaries
    print(f"\n{'Metric':<25} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 55)

    before_batches = before["batch_summaries"]
    after_batches = after["batch_summaries"]

    metrics = [
        ("Schema accuracy", "avg_schema_accuracy"),
        ("Execution success", "execution_success_rate"),
        ("Field correctness", "avg_field_score"),
    ]

    for label, key in metrics:
        b_vals = [b.get(key, 0) for b in before_batches]
        a_vals = [a.get(key, 0) for a in after_batches]
        b_avg = sum(b_vals) / len(b_vals) if b_vals else 0
        a_avg = sum(a_vals) / len(a_vals) if a_vals else 0
        delta = a_avg - b_avg
        arrow = "+" if delta > 0 else ""
        print(f"{label:<25} {b_avg:>9.1%} {a_avg:>9.1%} {arrow}{delta:>8.1%}")

    # Per-CSV regressions
    before_by_csv = {r["csv_file"]: r for r in before["individual_results"]}
    after_by_csv = {r["csv_file"]: r for r in after["individual_results"]}

    regressions = []
    improvements = []
    fixed_exec = []

    for csv_file in sorted(before_by_csv.keys()):
        if csv_file not in after_by_csv:
            continue
        b = before_by_csv[csv_file]
        a = after_by_csv[csv_file]

        b_exec = b.get("exec_score", {}).get("success", False)
        a_exec = a.get("exec_score", {}).get("success", False)

        if not b_exec and a_exec:
            fixed_exec.append(csv_file)
        elif b_exec and not a_exec:
            regressions.append((csv_file, "execution broke"))

        b_fields = b.get("field_score", {}).get("score", 0)
        a_fields = a.get("field_score", {}).get("score", 0)

        if a_fields < b_fields - 0.05 and a_exec:
            regressions.append((csv_file, f"fields {b_fields:.0%}→{a_fields:.0%}"))
        elif a_fields > b_fields + 0.05:
            improvements.append((csv_file, f"fields {b_fields:.0%}→{a_fields:.0%}"))

    print(f"\n--- Execution fixes: {len(fixed_exec)} CSVs recovered ---")
    for csv in fixed_exec[:10]:
        print(f"  {csv}")
    if len(fixed_exec) > 10:
        print(f"  ... and {len(fixed_exec) - 10} more")

    print(f"\n--- Improvements: {len(improvements)} CSVs ---")
    for csv, detail in improvements[:10]:
        print(f"  {csv}: {detail}")

    if regressions:
        print(f"\n--- REGRESSIONS: {len(regressions)} CSVs ---")
        for csv, detail in regressions:
            print(f"  {csv}: {detail}")
    else:
        print(f"\n--- No regressions detected ---")

    print(f"\n{'='*70}")


def main():
    eval_files = sorted(os.listdir(EVAL_DIR))
    eval_files = [f for f in eval_files if f.endswith(".json")]

    if len(sys.argv) == 3:
        before_path = os.path.join(EVAL_DIR, sys.argv[1])
        after_path = os.path.join(EVAL_DIR, sys.argv[2])
    elif len(eval_files) >= 2:
        before_path = os.path.join(EVAL_DIR, eval_files[-2])
        after_path = os.path.join(EVAL_DIR, eval_files[-1])
    else:
        print("Need at least 2 evaluation runs to compare.")
        return

    before = load_eval(before_path)
    after = load_eval(after_path)
    compare(before, after, os.path.basename(before_path), os.path.basename(after_path))


if __name__ == "__main__":
    main()
