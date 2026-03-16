"""
Trace analyzer for self-improvement.

Queries LangSmith for failure patterns, clusters errors, and generates
improvement rules that hooks inject into future runs.

Usage:
  uv run analyze_traces.py                      # analyze all traced runs
  uv run analyze_traces.py --min-failures 3     # only generate rules for patterns seen 3+ times

The analysis step is designed to be triggered:
  - Manually after evaluation
  - Automatically by agent.py every N runs (configured via analyze_every_n_runs)
"""

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime

from dotenv import load_dotenv
from langsmith import Client
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

RULES_FILE = "rules/improvement_rules.json"
PROJECT_NAME = "self-improving-agent"
AGENT_OUTPUT_DIR = "results/agent_outputs"

ls_client = Client()
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, max_tokens=2048)


# ---------------------------------------------------------------------------
# Step 1: Collect failure data from agent outputs + LangSmith feedback
# ---------------------------------------------------------------------------
def collect_failures(agent_output_dir: str = AGENT_OUTPUT_DIR) -> list[dict]:
    """Gather all execution failures from agent outputs."""
    failures = []

    for fname in sorted(os.listdir(agent_output_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(agent_output_dir, fname)) as f:
            data = json.load(f)

        if not data.get("error"):
            continue

        failures.append({
            "csv_file": data["csv_file"],
            "error": data["error"],
            "generated_code": data.get("generated_code", ""),
            "column_mapping": data.get("column_mapping", {}),
            "sample_rows": data.get("sample_rows", [])[:2],
            "run_id": data.get("run_id", ""),
        })

    return failures


# ---------------------------------------------------------------------------
# Step 2: Cluster errors into patterns
# ---------------------------------------------------------------------------
def cluster_errors(failures: list[dict]) -> dict[str, list[dict]]:
    """Group failures by error pattern."""
    clusters = {}

    for f in failures:
        err = f["error"]
        # Extract the core error pattern
        if "infer_datetime_format" in err:
            key = "deprecated_api: infer_datetime_format"
        elif "module 'datetime' has no attribute 'strptime'" in err:
            key = "namespace: datetime.strptime"
        elif "could not convert string to float" in err:
            key = "unclean_data: salary_string_not_cleaned"
        elif "doesn't match format" in err:
            key = "wrong_format: hardcoded_date_format"
        else:
            # Generic bucket — extract the exception type
            key = f"other: {err.split(':')[0] if ':' in err else err[:60]}"

        if key not in clusters:
            clusters[key] = []
        clusters[key].append(f)

    return clusters


# ---------------------------------------------------------------------------
# Step 3: Generate improvement rules using LLM
# ---------------------------------------------------------------------------
def generate_rules(clusters: dict[str, list[dict]], min_failures: int = 2) -> list[dict]:
    """Use LLM to generate improvement rules from error clusters."""

    # Filter to patterns that occur enough times to be worth a rule
    significant = {k: v for k, v in clusters.items() if len(v) >= min_failures}

    if not significant:
        print("  No error patterns meet the minimum frequency threshold.")
        return []

    # Build a summary for the LLM
    cluster_summary = []
    for pattern, failures in significant.items():
        examples = []
        for f in failures[:3]:  # max 3 examples per pattern
            examples.append({
                "csv_file": f["csv_file"],
                "error": f["error"][:200],
                "code_snippet": f["generated_code"][:300],
            })
        cluster_summary.append({
            "pattern": pattern,
            "count": len(failures),
            "examples": examples,
        })

    response = llm.invoke([
        SystemMessage(content="""You are an expert at analyzing code generation failures and creating improvement rules.

Given a set of error patterns from a code generation agent, generate concrete rules that will prevent these errors in future code generation.

Each rule should be:
- A clear, actionable instruction for a code-generating LLM
- Specific enough to prevent the exact error pattern
- General enough to cover variations of the same issue

Return a JSON array of rule objects, each with:
- "id": a short snake_case identifier
- "rule": the instruction text to inject into the code generation prompt
- "pattern": which error pattern this addresses
- "severity": "high" (causes crashes) or "medium" (causes bad output)

Return ONLY the JSON array, no markdown fences."""),
        HumanMessage(content=f"""Here are the error patterns from {sum(len(v) for v in significant.values())} failed runs:

{json.dumps(cluster_summary, indent=2, default=str)}

Generate improvement rules to prevent these errors."""),
    ])

    try:
        rules = json.loads(
            response.content.strip()
            .removeprefix("```json").removesuffix("```").strip()
        )
        return rules
    except json.JSONDecodeError:
        print(f"  Warning: could not parse LLM response as JSON")
        return []


# ---------------------------------------------------------------------------
# Step 4: Merge new rules into the rules file
# ---------------------------------------------------------------------------
def merge_rules(new_rules: list[dict]) -> int:
    """Merge new rules into the rules file, avoiding duplicates."""
    os.makedirs(os.path.dirname(RULES_FILE), exist_ok=True)
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE) as f:
            data = json.load(f)
    else:
        data = {"rules": [], "metadata": {"last_analysis": "", "total_runs_analyzed": 0, "version": 0}}

    existing_ids = {r["id"] for r in data["rules"]}
    added = 0

    for rule in new_rules:
        if rule["id"] not in existing_ids:
            rule["added_at"] = datetime.now().isoformat()
            data["rules"].append(rule)
            existing_ids.add(rule["id"])
            added += 1

    data["metadata"]["last_analysis"] = datetime.now().isoformat()
    data["metadata"]["total_runs_analyzed"] += len(os.listdir(AGENT_OUTPUT_DIR))
    data["metadata"]["version"] += 1

    with open(RULES_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return added


# ---------------------------------------------------------------------------
# Step 5: Validate rules by comparing composite scores
# ---------------------------------------------------------------------------
VALIDATION_CSVS = [f"test_{i:03d}.csv" for i in range(10)]  # test_000 through test_009


def validate_rules() -> bool:
    """Run agent on 10 sample CSVs with new rules and compare composite score.

    If the composite score drops, reverts to the previous rules file.
    Returns True if rules were kept, False if discarded.
    """
    from agent import run_migration
    from evaluate import compute_composite_for_csvs

    # Compute baseline composite score from current agent outputs (before re-running)
    baseline_score = compute_composite_for_csvs(VALIDATION_CSVS)

    # Backup current rules file
    rules_backup = RULES_FILE + ".backup"
    if os.path.exists(RULES_FILE):
        shutil.copy2(RULES_FILE, rules_backup)

    # Re-run agent on the 10 validation CSVs with the new rules
    print(f"\n  [validate] Running agent on {len(VALIDATION_CSVS)} validation CSVs...")
    for csv_file in VALIDATION_CSVS:
        csv_path = os.path.join("data/test", csv_file)
        if os.path.exists(csv_path):
            try:
                run_migration(csv_path)
            except Exception as e:
                print(f"    [validate] Error on {csv_file}: {e}")

    # Compute new composite score
    new_score = compute_composite_for_csvs(VALIDATION_CSVS)
    delta = new_score - baseline_score

    if delta >= 0:
        print(f"  [validate] Rules validated: composite {baseline_score:.2f} → {new_score:.2f} "
              f"({delta:+.2f}) — KEEPING")
        # Clean up backup
        if os.path.exists(rules_backup):
            os.remove(rules_backup)
        return True
    else:
        print(f"  [validate] Rules validated: composite {baseline_score:.2f} → {new_score:.2f} "
              f"({delta:+.2f}) — DISCARDING")
        # Revert to previous rules
        if os.path.exists(rules_backup):
            shutil.move(rules_backup, RULES_FILE)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_analysis(min_failures: int = 2, agent_output_dir: str = AGENT_OUTPUT_DIR):
    print(f"{'='*60}")
    print("Self-Improvement: Trace Analysis")
    print(f"{'='*60}")

    # Step 1: Collect failures
    failures = collect_failures(agent_output_dir=agent_output_dir)
    print(f"\nTotal execution failures: {len(failures)}")

    if not failures:
        print("No failures to analyze. Skipping.")
        return

    # Step 2: Cluster
    clusters = cluster_errors(failures)
    print(f"Error patterns found: {len(clusters)}")
    for pattern, items in sorted(clusters.items(), key=lambda x: -len(x[1])):
        print(f"  [{len(items):2d}x] {pattern}")

    # Step 3: Generate rules
    print(f"\nGenerating improvement rules (min_failures={min_failures})...")
    new_rules = generate_rules(clusters, min_failures=min_failures)
    print(f"Rules generated: {len(new_rules)}")

    if new_rules:
        for r in new_rules:
            print(f"  [{r['id']}] {r['rule'][:80]}")

    # Step 4: Merge
    added = merge_rules(new_rules)
    print(f"\nNew rules added to {RULES_FILE}: {added}")

    # Show current state
    with open(RULES_FILE) as f:
        data = json.load(f)
    print(f"Total rules in store: {len(data['rules'])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze traces and generate improvement rules.")
    parser.add_argument("--min-failures", type=int, default=2,
                        help="Minimum failures for a pattern to generate a rule (default: 2)")
    args = parser.parse_args()
    run_analysis(min_failures=args.min_failures)
