# Self-Improving Data Migration Agent

A LangGraph agent that takes messy CSV employee data and maps it to a standardized 6-field HR schema. It learns from its own failures using LangSmith traces, generating rules that get injected back into future prompts.

## Setup

```bash
cp .env.example .env   # add ANTHROPIC_API_KEY and LANGSMITH_API_KEY
uv sync
```

## Quick Test (single file)

A sample CSV (`data/test/test_100.csv`) is included in the repo for quick testing.

```bash
# 1. Run the baseline agent (no rules)
uv run agent_status_quo.py data/test/test_100.csv

# 2. Run with LangSmith tracing
uv run agent_status_quo_traced.py data/test/test_100.csv

# 3. Analyze failures and generate rules (min_failures=1 for single-file runs)
uv run python -c "from analyze_traces import run_analysis; run_analysis(min_failures=1, agent_output_dir='results/agent_status_quo')"

# 4. Check generated rules
cat rules/improvement_rules.json | python -m json.tool

# 5. Run the improved agent with rules
uv run agent_improved.py data/test/test_100.csv
```

Compare `results/agent_status_quo/test_100.json` vs `results/agent_improved/test_100.json` to see the improvement.

> **Note:** Rules are only generated from failures. If the CSV runs successfully, no rules will be added.

## Full Run (all 110 CSVs)

```bash
# Generate test data
uv run scripts/generate_datasets.py    # 100 base CSVs + ground truth
uv run scripts/add_hard_csvs.py        # 10 additional harder CSVs

# Run baseline
uv run agent_status_quo.py
uv run python -c "from evaluate import run_evaluation; run_evaluation(agent_output_dir='results/agent_status_quo')"

# Analyze and generate rules
uv run python -c "from analyze_traces import run_analysis; run_analysis(agent_output_dir='results/agent_status_quo')"

# Run improved agent
uv run agent_improved.py
uv run python -c "from evaluate import run_evaluation; run_evaluation(agent_output_dir='results/agent_improved')"
```

## Self-Improvement Loop

```text
run agent → evaluate → analyze failures → generate rules → inject into prompts
```

Rules are stored in `rules/improvement_rules.json` and injected into the code generation prompt on subsequent runs. A keep/discard gate compares scores before and after, reverting rules that make things worse.

## Evaluation Metrics

- **Schema accuracy** — does the column mapping match the golden dataset?
- **Execution success** — does the generated code run without errors?
- **Field correctness** — are output values valid (dates as YYYY-MM-DD, salaries as floats, emails with @)?

## Folder Structure

- `data/test/` — 110 synthetic test CSVs
- `data/ground_truth.json` — golden dataset with correct mappings
- `results/agent_status_quo/` — baseline agent outputs
- `results/agent_improved/` — improved agent outputs
- `results/evaluations/` — timestamped evaluation results
- `rules/` — learned improvement rules
- `scripts/` — data generation and experiment utilities
