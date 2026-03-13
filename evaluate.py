"""
Batch evaluation runner for the data migration agent.

Processes 100 CSVs in 5 batches of 20. After each batch:
  1. Scores all runs automatically (schema validation + pattern matching + string matching)
  2. Records scores as LangSmith feedback
  3. Adds confirmed-correct runs to a LangSmith dataset for future few-shot retrieval
  4. Re-indexes the dataset

Saves all results to results/evaluation_results.json.
"""

import argparse
import json
import os
import re
import uuid
from langsmith import Client
from agent import run_migration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 20
NUM_BATCHES = 5
GROUND_TRUTH = "data/ground_truth.json"
TEST_DIR = "data/test"
RESULTS_FILE = "results/evaluation_results.json"
DATASET_NAME = "migration-few-shot-examples"

ls_client = Client()

# ---------------------------------------------------------------------------
# Scoring Layer 1: Pattern matching (column mapping correctness)
# ---------------------------------------------------------------------------
def score_mapping(predicted: dict, ground_truth: dict) -> dict:
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
        "correct": [list(p) for p in correct],
        "incorrect": [list(p) for p in incorrect],
        "missing": [list(p) for p in missing],
    }


# ---------------------------------------------------------------------------
# Scoring Layer 2: Execution success
# ---------------------------------------------------------------------------
def score_execution(result: dict) -> dict:
    """Did the generated code execute without error?"""
    has_error = bool(result.get("error"))
    has_data = bool(result.get("transformed_data"))
    return {
        "success": not has_error and has_data,
        "error": result.get("error", ""),
    }


# ---------------------------------------------------------------------------
# Scoring Layer 3: Output quality (string matching on transformed data)
# ---------------------------------------------------------------------------
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def score_output_quality(result: dict) -> dict:
    """String-level checks on the actual transformed data."""
    data = result.get("transformed_data", [])
    if not data:
        return {"quality_score": 0.0, "issues": ["no output data"]}

    issues = []
    checks_passed = 0
    checks_total = 0

    for i, row in enumerate(data):
        # full_name: should look like a name (has a space, no @)
        name = row.get("full_name", "")
        checks_total += 1
        if name and " " in str(name) and "@" not in str(name):
            checks_passed += 1
        elif name:
            issues.append(f"row {i}: full_name '{name}' looks suspicious")

        # email: should contain @
        email = row.get("email", "")
        checks_total += 1
        if email and "@" in str(email):
            checks_passed += 1
        elif email:
            issues.append(f"row {i}: email '{email}' missing @")

        # start_date: should match YYYY-MM-DD
        date = row.get("start_date", "")
        checks_total += 1
        if date and DATE_PATTERN.match(str(date)):
            checks_passed += 1
        elif date:
            issues.append(f"row {i}: start_date '{date}' not YYYY-MM-DD")

        # salary_usd: should be a number in reasonable range
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

    quality_score = checks_passed / checks_total if checks_total > 0 else 0.0
    return {"quality_score": quality_score, "issues": issues}


# ---------------------------------------------------------------------------
# LangSmith feedback
# ---------------------------------------------------------------------------
def record_feedback(run_id, mapping_score, exec_score, quality_score):
    """Attach all scores as LangSmith feedback on the run."""
    try:
        ls_client.create_feedback(
            run_id=run_id,
            key="mapping_accuracy",
            score=mapping_score["accuracy"],
        )
        ls_client.create_feedback(
            run_id=run_id,
            key="execution_success",
            score=1.0 if exec_score["success"] else 0.0,
        )
        ls_client.create_feedback(
            run_id=run_id,
            key="output_quality",
            score=quality_score["quality_score"],
        )
    except Exception as e:
        print(f"    Warning: feedback recording failed: {e}")


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------
def ensure_dataset():
    """Create the few-shot dataset if it doesn't exist. Return dataset_id."""
    datasets = list(ls_client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        return datasets[0].id
    dataset = ls_client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Confirmed-correct CSV column mappings for few-shot injection",
    )
    return dataset.id


def add_confirmed_correct(dataset_id, result, ground_truth_entry):
    """Add a confirmed-correct mapping to the LangSmith dataset."""
    ls_client.create_examples(
        dataset_id=dataset_id,
        examples=[{
            "inputs": {
                "columns": json.dumps(result["raw_columns"]),
                "sample_data": json.dumps(result["sample_rows"][:2], default=str),
            },
            "outputs": {
                "column_mapping": ground_truth_entry["column_mapping"],
                "generated_code": result.get("generated_code", ""),
            },
            "metadata": {
                "csv_file": ground_truth_entry["csv_file"],
                "challenges": ground_truth_entry["challenges"],
            },
        }],
    )


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------
def run_evaluation(num_batches: int = NUM_BATCHES):
    with open(GROUND_TRUTH) as f:
        all_gt = json.load(f)

    dataset_id = ensure_dataset()
    all_results = []
    batch_summaries = []

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        batch_gt = all_gt[start:end]

        print(f"\n{'='*70}")
        print(f"BATCH {batch_idx + 1}/{num_batches}  (CSVs {start}–{end - 1})")
        print(f"{'='*70}")

        batch_mapping_sum = 0.0
        batch_exec_ok = 0
        batch_quality_sum = 0.0
        batch_confirmed = 0
        batch_results = []

        for i, gt_entry in enumerate(batch_gt):
            csv_path = os.path.join(TEST_DIR, gt_entry["csv_file"])
            global_idx = start + i
            run_id = uuid.uuid4()

            print(f"  [{global_idx:03d}] {gt_entry['csv_file']} ", end="", flush=True)

            # Run agent
            try:
                result = run_migration(csv_path, run_id=run_id)
            except Exception as e:
                result = {
                    "csv_path": csv_path,
                    "raw_columns": [],
                    "sample_rows": [],
                    "column_mapping": {},
                    "generated_code": "",
                    "transformed_data": [],
                    "validation_result": {"valid": False, "issues": [str(e)]},
                    "error": str(e),
                    "_run_id": run_id,
                }

            # Score (3 layers)
            m_score = score_mapping(result.get("column_mapping", {}), gt_entry["column_mapping"])
            e_score = score_execution(result)
            q_score = score_output_quality(result)

            is_confirmed = (
                m_score["accuracy"] == 1.0
                and e_score["success"]
                and q_score["quality_score"] >= 0.8
            )

            # Print inline
            status = "OK" if is_confirmed else "FAIL"
            print(f"map={m_score['accuracy']:.0%}  exec={'OK' if e_score['success'] else 'ERR'}  "
                  f"quality={q_score['quality_score']:.0%}  [{status}]")

            # Record to LangSmith
            record_feedback(run_id, m_score, e_score, q_score)

            # If confirmed correct, add to few-shot dataset
            if is_confirmed:
                add_confirmed_correct(dataset_id, result, gt_entry)
                batch_confirmed += 1

            # Accumulate
            batch_mapping_sum += m_score["accuracy"]
            if e_score["success"]:
                batch_exec_ok += 1
            batch_quality_sum += q_score["quality_score"]

            batch_results.append({
                "csv_file": gt_entry["csv_file"],
                "run_id": str(run_id),
                "mapping_score": m_score,
                "exec_score": e_score,
                "quality_score": q_score,
                "confirmed_correct": is_confirmed,
                "challenges": gt_entry["challenges"],
                "date_format": gt_entry["date_format"],
                "salary_format": gt_entry["salary_format"],
            })

        # Re-index dataset for next batch
        try:
            ls_client.index_dataset(dataset_id=dataset_id)
            print(f"\n  Dataset re-indexed (+{batch_confirmed} examples)")
        except Exception as e:
            print(f"\n  Warning: dataset indexing failed: {e}")

        # Batch summary
        summary = {
            "batch": batch_idx + 1,
            "csv_range": f"{start}–{end - 1}",
            "avg_mapping_accuracy": batch_mapping_sum / BATCH_SIZE,
            "execution_success_rate": batch_exec_ok / BATCH_SIZE,
            "avg_output_quality": batch_quality_sum / BATCH_SIZE,
            "confirmed_correct_count": batch_confirmed,
        }
        batch_summaries.append(summary)
        all_results.extend(batch_results)

        print(f"\n  Batch {batch_idx + 1}/{num_batches} Summary:")
        print(f"    Mapping accuracy:    {summary['avg_mapping_accuracy']:.1%}")
        print(f"    Execution success:   {summary['execution_success_rate']:.1%}")
        print(f"    Output quality:      {summary['avg_output_quality']:.1%}")
        print(f"    Confirmed correct:   {batch_confirmed}/{BATCH_SIZE}")

    # Cumulative few-shot count
    cumulative = 0
    for s in batch_summaries:
        cumulative += s["confirmed_correct_count"]
        s["cumulative_few_shot_examples"] = cumulative

    # Save
    output = {"batch_summaries": batch_summaries, "individual_results": all_results}
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"DONE — results saved to {RESULTS_FILE}")
    print(f"{'='*70}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluation runner for the data migration agent.")
    parser.add_argument("--batches", type=int, default=NUM_BATCHES,
                        help=f"Number of batches to run (default: {NUM_BATCHES}). Use 1 for a quick test.")
    args = parser.parse_args()
    run_evaluation(num_batches=args.batches)
