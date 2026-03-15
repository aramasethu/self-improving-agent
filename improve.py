"""
One-line improvement cycle for the self-improving data migration agent.

Wraps: run agent → evaluate → analyze traces → validate rules into a single call.

Usage:
  uv run improve.py                          # run one improvement cycle
  uv run python -c "from improve import improve; r = improve(); print(r)"
"""

import json
import os

from agent import run_all, get_current_iteration, AGENT_OUTPUT_DIR
from evaluate import run_evaluation, compute_composite_for_csvs
from analyze_traces import run_analysis, validate_rules, VALIDATION_CSVS, RULES_FILE


def improve(test_dir="data/test", sample_size=10, verbose=True):
    """Run one improvement cycle:
    1. Run agent on all CSVs
    2. Evaluate results (composite score)
    3. Analyze traces for failure patterns
    4. Generate candidate rules
    5. Validate rules (keep/discard)
    6. Print summary: before score, after score, rules added/discarded

    Returns a result dict with iteration stats.
    """
    # Count rules before
    rules_before = 0
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE) as f:
            rules_before = len(json.load(f).get("rules", []))

    # Composite score before (from existing agent outputs)
    composite_before = compute_composite_for_csvs(VALIDATION_CSVS)

    # Step 1: Run agent on all CSVs
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 1: Running agent on all CSVs")
        print("=" * 60)
    run_all(test_dir=test_dir)

    # Step 2: Evaluate
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 2: Evaluating results")
        print("=" * 60)
    run_evaluation()

    # Step 3: Analyze traces and generate rules
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 3: Analyzing traces")
        print("=" * 60)
    run_analysis(min_failures=1)

    # Step 4: Validate rules (keep or discard)
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 4: Validating rules")
        print("=" * 60)
    kept = validate_rules()

    # Composite score after
    composite_after = compute_composite_for_csvs(VALIDATION_CSVS)

    # Count rules after
    rules_after = 0
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE) as f:
            rules_after = len(json.load(f).get("rules", []))

    rules_added = max(0, rules_after - rules_before)
    rules_discarded = 0 if kept else max(0, rules_added)
    if not kept:
        rules_added = 0

    # Total LLM calls across all outputs
    llm_calls_total = 0
    if os.path.exists(AGENT_OUTPUT_DIR):
        for fname in os.listdir(AGENT_OUTPUT_DIR):
            if fname.endswith(".json"):
                with open(os.path.join(AGENT_OUTPUT_DIR, fname)) as f:
                    data = json.load(f)
                llm_calls_total += data.get("llm_call_count", 0)

    iteration = get_current_iteration()

    result = {
        "iteration": iteration,
        "composite_before": round(composite_before, 4),
        "composite_after": round(composite_after, 4),
        "rules_added": rules_added,
        "rules_discarded": rules_discarded,
        "llm_calls_total": llm_calls_total,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("IMPROVEMENT CYCLE COMPLETE")
        print("=" * 60)
        print(f"  Iteration:        {result['iteration']}")
        print(f"  Composite before: {result['composite_before']:.2%}")
        print(f"  Composite after:  {result['composite_after']:.2%}")
        delta = result['composite_after'] - result['composite_before']
        print(f"  Delta:            {delta:+.2%}")
        print(f"  Rules added:      {result['rules_added']}")
        print(f"  Rules discarded:  {result['rules_discarded']}")
        print(f"  Total LLM calls:  {result['llm_calls_total']}")

    return result


if __name__ == "__main__":
    improve()
