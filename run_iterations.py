"""
Run 3 iterations of the self-improving agent and evaluate each.

Iteration 1: Baseline — no rules, fresh start
Iteration 2: Analyze traces from iteration 1, then run with learned rules
Iteration 3: Analyze traces from iteration 2, accumulate more rules, then run

Each iteration is tagged in LangSmith (iteration-1, iteration-2, iteration-3).
Evaluation results are saved with timestamps so all 3 can be compared afterward.

Usage:
  uv run run_iterations.py
"""

import json
import os
import shutil

RULES_FILE = "rules/improvement_rules.json"
ITERATION_COUNTER_FILE = "rules/.iteration_counter"
AGENT_OUTPUT_DIR = "results/agent_outputs"
AGENT_OUTPUT_BACKUP_DIR = "results/agent_outputs_iter_{}"

EMPTY_RULES = {
    "rules": [],
    "metadata": {
        "last_analysis": "",
        "total_runs_analyzed": 0,
        "version": 0,
    },
}


def reset_rules():
    """Clear all rules for a fresh baseline."""
    os.makedirs(os.path.dirname(RULES_FILE), exist_ok=True)
    with open(RULES_FILE, "w") as f:
        json.dump(EMPTY_RULES, f, indent=2)
    print("  Rules reset to empty.")


def reset_iteration_counter():
    """Reset iteration counter so we start at 1."""
    os.makedirs(os.path.dirname(ITERATION_COUNTER_FILE), exist_ok=True)
    with open(ITERATION_COUNTER_FILE, "w") as f:
        f.write("0")


def backup_outputs(iteration: int):
    """Copy agent outputs to a backup dir for this iteration."""
    backup_dir = AGENT_OUTPUT_BACKUP_DIR.format(iteration)
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    if os.path.exists(AGENT_OUTPUT_DIR):
        shutil.copytree(AGENT_OUTPUT_DIR, backup_dir)
        print(f"  Outputs backed up to {backup_dir}/")


def show_rules_count():
    """Print how many rules are currently loaded."""
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE) as f:
            data = json.load(f)
        print(f"  Rules in store: {len(data['rules'])}")
    else:
        print("  Rules in store: 0")


def main():
    from agent import run_all
    from evaluate import run_evaluation
    from analyze_traces import run_analysis, validate_rules

    print("=" * 70)
    print("ITERATION RUNNER — 3 iterations with self-improvement")
    print("=" * 70)

    # Reset for clean baseline
    reset_rules()
    reset_iteration_counter()

    for iteration in range(1, 4):
        print(f"\n{'#' * 70}")
        print(f"# ITERATION {iteration}")
        print(f"{'#' * 70}")

        if iteration > 1:
            # Analyze traces from previous iteration's outputs
            print(f"\n--- Analyzing traces from iteration {iteration - 1} ---")
            show_rules_count()
            run_analysis(min_failures=1)
            show_rules_count()

            # Validate: keep rules only if they don't hurt composite score
            print(f"\n--- Validating rules ---")
            kept = validate_rules()
            if not kept:
                print("  Bad rules discarded — continuing with previous rules.")
            show_rules_count()

        # Run agent on all CSVs
        print(f"\n--- Running agent (iteration {iteration}) ---")
        show_rules_count()
        run_all()

        # Backup outputs before they get overwritten next iteration
        backup_outputs(iteration)

        # Evaluate
        print(f"\n--- Evaluating iteration {iteration} ---")
        run_evaluation()

    print(f"\n{'#' * 70}")
    print("# ALL ITERATIONS COMPLETE")
    print(f"{'#' * 70}")
    print("\nTo compare iterations, run:")
    print("  uv run compare_runs.py <eval_1.json> <eval_2.json>")
    print("\nEvaluation files are in results/evaluations/ (sorted by timestamp).")
    print("Agent output backups are in results/agent_outputs_iter_1/, _iter_2/, _iter_3/")


if __name__ == "__main__":
    main()
