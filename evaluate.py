"""
Evaluation script for the data migration agent.

Reads pre-generated agent outputs from results/agent_outputs/ and scores them
against the golden dataset at data/ground_truth.json.

Metrics:
  1. Schema correctness — does the column mapping match ground truth?
  2. Execution success — did the generated code run without error?
  3. Field value correctness — are transformed values valid (regex/string matching)?

Usage:
  uv run evaluate.py                  # evaluate all 100 CSVs
  uv run evaluate.py --batches 1      # evaluate first batch only (20 CSVs)
"""

import argparse
import json
import os
import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 20
NUM_BATCHES = 5
GROUND_TRUTH = "data/ground_truth.json"
AGENT_OUTPUT_DIR = "results/agent_outputs"
EVAL_RESULTS_DIR = "results/evaluations"

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# ---------------------------------------------------------------------------
# Metric 1: Schema correctness (column mapping)
# ---------------------------------------------------------------------------
def score_schema(predicted: dict, ground_truth: dict) -> dict:
    """Compare predicted vs ground truth column mappings as sets of (source, target) pairs."""
    gt_pairs = {(src.strip(), tgt) for src, tgt in ground_truth.items()}
    pred_pairs = {(src.strip(), tgt) for src, tgt in predicted.items()}

    correct = gt_pairs & pred_pairs
    incorrect = pred_pairs - gt_pairs
    missing = gt_pairs - pred_pairs

    total = len(gt_pairs)
    accuracy = len(correct) / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": [list(p) for p in sorted(correct)],
        "incorrect": [list(p) for p in sorted(incorrect)],
        "missing": [list(p) for p in sorted(missing)],
    }


# ---------------------------------------------------------------------------
# Metric 2: Execution success
# ---------------------------------------------------------------------------
def score_execution(agent_result: dict) -> dict:
    """Did the generated code execute without error and produce output?"""
    error = agent_result.get("error", "")
    has_data = bool(agent_result.get("transformed_data"))
    success = not error and has_data
    return {
        "success": success,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Metric 3: Field value correctness (regex / string matching)
# ---------------------------------------------------------------------------
def score_fields(agent_result: dict) -> dict:
    """Check transformed field values using regex and string matching."""
    data = agent_result.get("transformed_data", [])
    if not data:
        return {"score": 0.0, "issues": ["no output data"]}

    issues = []
    checks_passed = 0
    checks_total = 0

    for i, row in enumerate(data):
        # full_name: has a space (first + last), no @ symbol
        name = row.get("full_name", "")
        checks_total += 1
        if name and " " in str(name) and "@" not in str(name):
            checks_passed += 1
        elif name:
            issues.append(f"row {i}: full_name '{name}' looks invalid")

        # email: contains @
        email = row.get("email", "")
        checks_total += 1
        if email and "@" in str(email):
            checks_passed += 1
        elif email:
            issues.append(f"row {i}: email '{email}' missing @")

        # department: non-empty string
        dept = row.get("department", "")
        checks_total += 1
        if dept and isinstance(dept, str) and len(dept.strip()) > 0:
            checks_passed += 1
        else:
            issues.append(f"row {i}: department is empty or invalid")

        # start_date: matches YYYY-MM-DD
        date = row.get("start_date", "")
        checks_total += 1
        if date and DATE_PATTERN.match(str(date)):
            checks_passed += 1
        elif date:
            issues.append(f"row {i}: start_date '{date}' not YYYY-MM-DD")

        # salary_usd: valid float in reasonable range
        salary = row.get("salary_usd")
        checks_total += 1
        try:
            sal_val = float(salary)
            if 30000 <= sal_val <= 500000:
                checks_passed += 1
            else:
                issues.append(f"row {i}: salary_usd {sal_val} outside 30k-500k range")
        except (TypeError, ValueError):
            issues.append(f"row {i}: salary_usd '{salary}' not a valid number")

        # manager_name: non-empty string with a space (first + last)
        mgr = row.get("manager_name", "")
        checks_total += 1
        if mgr and " " in str(mgr):
            checks_passed += 1
        elif mgr:
            issues.append(f"row {i}: manager_name '{mgr}' looks invalid")

    score = checks_passed / checks_total if checks_total > 0 else 0.0
    return {"score": score, "issues": issues}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation(num_batches: int = NUM_BATCHES):
    with open(GROUND_TRUTH) as f:
        all_gt = json.load(f)

    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []
    batch_summaries = []

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        batch_gt = all_gt[start:end]

        print(f"\n{'='*70}")
        print(f"BATCH {batch_idx + 1}/{num_batches}  (CSVs {start}–{end - 1})")
        print(f"{'='*70}")

        batch_schema_sum = 0.0
        batch_exec_ok = 0
        batch_field_sum = 0.0
        batch_results = []

        for i, gt_entry in enumerate(batch_gt):
            global_idx = start + i
            csv_file = gt_entry["csv_file"]
            base_name = os.path.splitext(csv_file)[0]
            agent_output_path = os.path.join(AGENT_OUTPUT_DIR, f"{base_name}.json")

            # Load agent output
            if not os.path.exists(agent_output_path):
                print(f"  [{global_idx:03d}] {csv_file}  MISSING — run agent.py first")
                batch_results.append({
                    "csv_file": csv_file,
                    "schema_score": {"accuracy": 0.0, "correct": [], "incorrect": [], "missing": []},
                    "exec_score": {"success": False, "error": "agent output not found"},
                    "field_score": {"score": 0.0, "issues": ["agent output not found"]},
                    "challenges": gt_entry["challenges"],
                    "error": "agent output not found",
                })
                continue

            with open(agent_output_path) as f:
                agent_result = json.load(f)

            # Score
            schema_score = score_schema(
                agent_result.get("column_mapping", {}),
                gt_entry["column_mapping"],
            )
            exec_score = score_execution(agent_result)
            field_score = score_fields(agent_result)

            exec_label = "OK" if exec_score["success"] else "ERR"
            print(f"  [{global_idx:03d}] {csv_file}  "
                  f"schema={schema_score['accuracy']:.0%}  "
                  f"exec={exec_label}  "
                  f"fields={field_score['score']:.0%}")

            batch_schema_sum += schema_score["accuracy"]
            if exec_score["success"]:
                batch_exec_ok += 1
            batch_field_sum += field_score["score"]

            batch_results.append({
                "csv_file": csv_file,
                "schema_score": schema_score,
                "exec_score": exec_score,
                "field_score": field_score,
                "challenges": gt_entry["challenges"],
                "error": agent_result.get("error", ""),
            })

        batch_size = len(batch_gt)
        summary = {
            "batch": batch_idx + 1,
            "csv_range": f"{start}–{end - 1}",
            "avg_schema_accuracy": batch_schema_sum / batch_size,
            "execution_success_rate": batch_exec_ok / batch_size,
            "avg_field_score": batch_field_sum / batch_size,
        }
        batch_summaries.append(summary)
        all_results.extend(batch_results)

        print(f"\n  Batch {batch_idx + 1}/{num_batches} Summary:")
        print(f"    Schema accuracy:     {summary['avg_schema_accuracy']:.1%}")
        print(f"    Execution success:   {summary['execution_success_rate']:.1%}")
        print(f"    Field correctness:   {summary['avg_field_score']:.1%}")

    # Save
    output = {"timestamp": timestamp, "batch_summaries": batch_summaries, "individual_results": all_results}
    eval_file = os.path.join(EVAL_RESULTS_DIR, f"eval_{timestamp}.json")
    with open(eval_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"DONE — results saved to {eval_file}")
    print(f"{'='*70}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent outputs against ground truth.")
    parser.add_argument("--batches", type=int, default=NUM_BATCHES,
                        help=f"Number of batches to evaluate (default: {NUM_BATCHES}). Use 1 for a quick test.")
    args = parser.parse_args()
    run_evaluation(num_batches=args.batches)
