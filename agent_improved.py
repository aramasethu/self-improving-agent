"""
Self-Improving Data Migration Agent
====================================
Builds on the status quo agent by adding:
  - LangSmith tracing (to capture failures)
  - Trace analysis (to find error patterns)
  - Rule generation (LLM writes rules from patterns)
  - Rule validation (keep only rules that help)
  - Rule injection (rules go into future prompts)

Usage:
  uv run agent_improved.py                          # run full improvement cycle
  uv run agent_improved.py data/test/test_019.csv   # run one CSV with current rules
  uv run agent_improved.py --improve                # run all + analyze + validate rules
"""

import json
import os
import sys
import uuid
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ---------------------------------------------------------------------------
# LangSmith tracing ON
# ---------------------------------------------------------------------------
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ.setdefault("LANGCHAIN_PROJECT", "self-improving-agent")

# ---------------------------------------------------------------------------
# Target schema
# ---------------------------------------------------------------------------
TARGET_SCHEMA = {
    "full_name": "Employee full name (string)",
    "email": "Work email address (string)",
    "department": "Department name (string)",
    "start_date": "Start date in YYYY-MM-DD format (string)",
    "salary_usd": "Annual salary in USD (float)",
    "manager_name": "Manager's full name (string)",
}

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    max_tokens=2048,
)

# ---------------------------------------------------------------------------
# Module-level state for tools
# ---------------------------------------------------------------------------
_current_df: pd.DataFrame | None = None
_result_df: pd.DataFrame | None = None
_llm_call_count: int = 0
MAX_LLM_CALLS_PER_RUN = 10
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Output + rules config
# ---------------------------------------------------------------------------
AGENT_OUTPUT_DIR = "results/agent_improved"
RULES_FILE = "rules/improvement_rules.json"

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
@tool
def read_csv_sample(file_path: str, num_rows: int = 5) -> str:
    """Read a CSV file and return column names + sample rows for inspection.

    Args:
        file_path: Path to the CSV file to read.
        num_rows: Number of sample rows to return (default 5).
    """
    global _current_df
    _current_df = pd.read_csv(file_path)
    sample = _current_df.head(num_rows)
    cols = _current_df.columns.tolist()
    return f"Columns: {cols}\n\nSample rows ({num_rows}):\n{sample.to_string(index=False)}"


@tool
def run_code(code: str) -> str:
    """Execute pandas transformation code against the loaded DataFrame.

    Args:
        code: Python/Pandas code to execute.
    """
    global _result_df
    if _current_df is None:
        return "Error: No DataFrame loaded. Call read_csv_sample first."
    exec_globals = {"pd": pd, "re": __import__("re"), "datetime": __import__("datetime")}
    local_vars = {"df": _current_df.copy()}
    try:
        exec(code, exec_globals, local_vars)
        _result_df = local_vars["result_df"]
        return f"Success. result_df shape: {_result_df.shape}\n\nFirst 3 rows:\n{_result_df.head(3).to_string(index=False)}"
    except Exception as e:
        return f"Error: {e}"


TOOLS = [read_csv_sample, run_code]
TOOL_MAP = {t.name: t for t in TOOLS}


def run_llm_with_tools(messages: list, tools: list | None = None) -> str:
    """Run LLM in a loop, executing tool calls until text response."""
    global _llm_call_count
    if tools is None:
        tools = TOOLS
    llm_with_tools = llm.bind_tools(tools)

    while True:
        if _llm_call_count >= MAX_LLM_CALLS_PER_RUN:
            return "Error: LLM call budget exhausted"
        _llm_call_count += 1
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.content
        for tc in response.tool_calls:
            tool_fn = TOOL_MAP[tc["name"]]
            result = tool_fn.invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
def load_rules() -> list[dict]:
    if not os.path.exists(RULES_FILE):
        return []
    with open(RULES_FILE) as f:
        return json.load(f).get("rules", [])


def format_rules_for_prompt(rules: list[dict]) -> str:
    if not rules:
        return ""
    lines = ["IMPORTANT - Learned rules from previous failures (you MUST follow these):"]
    for r in rules:
        lines.append(f"- {r['rule']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    csv_path: str
    raw_columns: list[str]
    sample_rows: list[dict]
    column_mapping: dict
    generated_code: str
    transformed_data: list[dict]
    validation_result: dict
    error: str
    retry_count: int
    retry_errors: list[str]
    llm_call_count: int


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def analyze_csv(state: AgentState) -> AgentState:
    """LLM calls read_csv_sample tool to inspect the CSV."""
    messages = [
        SystemMessage(content="You are a data migration agent. Use the read_csv_sample tool to inspect the CSV file. "
                      "After inspecting, briefly summarize what you see."),
        HumanMessage(content=f"Please inspect this CSV file: {state['csv_path']}"),
    ]
    run_llm_with_tools(messages, tools=[read_csv_sample])
    return {
        "raw_columns": _current_df.columns.tolist(),
        "sample_rows": _current_df.head(3).to_dict(orient="records"),
    }


def map_columns(state: AgentState) -> AgentState:
    global _llm_call_count
    if _llm_call_count >= MAX_LLM_CALLS_PER_RUN:
        return {"column_mapping": {}, "error": "LLM call budget exhausted"}

    target_desc = json.dumps(TARGET_SCHEMA, indent=2)
    source_cols = json.dumps(state["raw_columns"])
    sample = json.dumps(state["sample_rows"], indent=2, default=str)

    _llm_call_count += 1
    response = llm.invoke([
        SystemMessage(content=f"""You are a data migration expert. Map source CSV columns to a target schema.

Target schema:
{target_desc}

Rules:
- Map each target field to exactly one source column
- If a source column doesn't map to any target field, ignore it
- Return ONLY a JSON object mapping source column names to target field names
- Example: {{"emp_name": "full_name", "email_addr": "email"}}
"""),
        HumanMessage(content=f"""Source columns: {source_cols}

Sample data:
{sample}

Return the mapping as a JSON object. Nothing else."""),
    ])

    mapping = json.loads(response.content.strip().removeprefix("```json").removesuffix("```").strip())
    return {"column_mapping": mapping}


def generate_code(state: AgentState) -> AgentState:
    global _llm_call_count
    if _llm_call_count >= MAX_LLM_CALLS_PER_RUN:
        return {"generated_code": "", "error": "LLM call budget exhausted"}

    mapping = json.dumps(state["column_mapping"], indent=2)
    sample = json.dumps(state["sample_rows"], indent=2, default=str)
    target = json.dumps(TARGET_SCHEMA, indent=2)

    rules = load_rules()
    rules_text = format_rules_for_prompt(rules)

    error_context = ""
    retry_errors = state.get("retry_errors", [])
    if retry_errors:
        error_context = "\n\nPREVIOUS ATTEMPTS FAILED with these errors (do NOT repeat them):\n"
        for i, err in enumerate(retry_errors):
            error_context += f"  Attempt {i+1}: {err}\n"
        error_context += "\nFix the code to avoid these specific errors."

    system_prompt = f"""You are a Python/Pandas expert. Generate code to transform a CSV dataframe.

The code must:
1. Assume `df` is already loaded as a pandas DataFrame
2. Rename columns according to the mapping
3. Convert the start_date column to YYYY-MM-DD format (handle various date formats)
4. Ensure salary_usd is a float
5. Select only the target schema columns
6. Assign the result to a variable called `result_df`

Target schema:
{target}

{rules_text}
{error_context}

IMPORTANT: Return ONLY the Python code, no markdown fences, no explanation."""

    _llm_call_count += 1
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Column mapping (source -> target): {mapping}

Sample data:
{sample}

Generate the transformation code."""),
    ])

    code = response.content.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()
    return {"generated_code": code}


def execute_code(state: AgentState) -> AgentState:
    global _result_df
    _result_df = None
    code = state["generated_code"]

    messages = [
        SystemMessage(content="You are a data migration agent. Use the run_code tool to execute the provided "
                      "transformation code. Report whether it succeeded or failed."),
        HumanMessage(content=f"Execute this transformation code:\n\n{code}"),
    ]
    result_text = run_llm_with_tools(messages, tools=[run_code])

    if _result_df is not None:
        return {"transformed_data": _result_df.to_dict(orient="records"), "error": ""}
    else:
        retry_count = state.get("retry_count", 0)
        retry_errors = list(state.get("retry_errors", []))
        retry_errors.append(result_text)
        return {
            "error": f"Code execution failed: {result_text}",
            "retry_count": retry_count,
            "retry_errors": retry_errors,
        }


def validate_output(state: AgentState) -> AgentState:
    if state.get("error"):
        return {"validation_result": {"valid": False, "error": state["error"]}}

    data = state["transformed_data"]
    expected_cols = set(TARGET_SCHEMA.keys())
    issues = []

    if not data:
        issues.append("No data rows produced")
    else:
        actual_cols = set(data[0].keys())
        missing = expected_cols - actual_cols
        if missing:
            issues.append(f"Missing columns: {missing}")
        for i, row in enumerate(data):
            if not row.get("start_date") or not isinstance(row["start_date"], str):
                issues.append(f"Row {i}: invalid start_date")
            if row.get("salary_usd") is not None:
                try:
                    float(row["salary_usd"])
                except (ValueError, TypeError):
                    issues.append(f"Row {i}: salary_usd not a valid number")

    return {"validation_result": {"valid": len(issues) == 0, "issues": issues, "row_count": len(data)}}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def should_retry(state: AgentState) -> str:
    if not state.get("error"):
        return "validate_output"
    retry_count = state.get("retry_count", 0)
    if retry_count < MAX_RETRIES and _llm_call_count < MAX_LLM_CALLS_PER_RUN:
        print(f"    [retry] {retry_count + 1}/{MAX_RETRIES} "
              f"(LLM: {_llm_call_count}/{MAX_LLM_CALLS_PER_RUN}): {state['error'][:80]}")
        return "retry_generate"
    return "validate_output"


def retry_generate(state: AgentState) -> AgentState:
    retry_count = state.get("retry_count", 0) + 1
    new_state = generate_code({**state, "retry_count": retry_count})
    new_state["retry_count"] = retry_count
    return new_state


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze_csv", analyze_csv)
    graph.add_node("map_columns", map_columns)
    graph.add_node("generate_code", generate_code)
    graph.add_node("execute_code", execute_code)
    graph.add_node("retry_generate", retry_generate)
    graph.add_node("validate_output", validate_output)

    graph.add_edge(START, "analyze_csv")
    graph.add_edge("analyze_csv", "map_columns")
    graph.add_edge("map_columns", "generate_code")
    graph.add_edge("generate_code", "execute_code")

    graph.add_conditional_edges("execute_code", should_retry, {
        "retry_generate": "retry_generate",
        "validate_output": "validate_output",
    })

    graph.add_edge("retry_generate", "execute_code")
    graph.add_edge("validate_output", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_migration(csv_path: str):
    global _llm_call_count
    _llm_call_count = 0

    print(f"Processing: {csv_path}")
    app = build_graph()
    run_id = uuid.uuid4()

    initial_state = {
        "csv_path": csv_path,
        "raw_columns": [],
        "sample_rows": [],
        "column_mapping": {},
        "generated_code": "",
        "transformed_data": [],
        "validation_result": {},
        "error": "",
        "retry_count": 0,
        "retry_errors": [],
        "llm_call_count": 0,
    }

    config = {
        "run_id": run_id,
        "run_name": f"migrate-{os.path.basename(csv_path)}",
        "metadata": {"csv_file": os.path.basename(csv_path)},
        "tags": ["improved"],
    }
    result = app.invoke(initial_state, config=config)

    os.makedirs(AGENT_OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(AGENT_OUTPUT_DIR, f"{base_name}.json")

    serializable = {
        "csv_file": os.path.basename(csv_path),
        "csv_path": csv_path,
        "raw_columns": result["raw_columns"],
        "sample_rows": result["sample_rows"],
        "column_mapping": result["column_mapping"],
        "generated_code": result["generated_code"],
        "transformed_data": result["transformed_data"],
        "validation_result": result["validation_result"],
        "error": result.get("error", ""),
        "run_id": str(run_id),
        "retry_count": result.get("retry_count", 0),
        "retry_errors": result.get("retry_errors", []),
        "llm_call_count": _llm_call_count,
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    valid = result.get("validation_result", {}).get("valid", False)
    status = "OK" if valid else "FAIL"
    print(f"  [{status}] Saved: {output_path} (LLM calls: {_llm_call_count})")

    return result


def run_all(test_dir: str = "data/test"):
    csv_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".csv"))

    rules = load_rules()
    print(f"Running improved agent on {len(csv_files)} CSVs from {test_dir}/")
    print(f"Rules loaded: {len(rules)}  |  Max retries: {MAX_RETRIES}  |  LLM budget: {MAX_LLM_CALLS_PER_RUN}")
    print(f"LangSmith tracing ON  →  project: self-improving-agent")
    print()

    for csv_file in csv_files:
        csv_path = os.path.join(test_dir, csv_file)
        try:
            run_migration(csv_path)
        except Exception as e:
            print(f"  [ERROR] {csv_file}: {e}")
            os.makedirs(AGENT_OUTPUT_DIR, exist_ok=True)
            base_name = os.path.splitext(csv_file)[0]
            output_path = os.path.join(AGENT_OUTPUT_DIR, f"{base_name}.json")
            with open(output_path, "w") as f:
                json.dump({
                    "csv_file": csv_file, "csv_path": csv_path,
                    "raw_columns": [], "sample_rows": [], "column_mapping": {},
                    "generated_code": "", "transformed_data": [],
                    "validation_result": {}, "error": str(e),
                    "run_id": "", "retry_count": 0, "retry_errors": [],
                    "llm_call_count": _llm_call_count,
                }, f, indent=2)

    print(f"\nDone. Results saved to {AGENT_OUTPUT_DIR}/")


def run_improve(test_dir: str = "data/test"):
    """Full improvement cycle: run all → analyze traces → validate rules → report."""
    from evaluate import run_evaluation, compute_composite_for_csvs
    from analyze_traces import run_analysis, validate_rules, VALIDATION_CSVS

    # Score before
    composite_before = compute_composite_for_csvs(
        VALIDATION_CSVS,
        agent_output_dir=AGENT_OUTPUT_DIR,
    ) if os.path.exists(AGENT_OUTPUT_DIR) else 0.0

    rules_before = len(load_rules())

    # Step 1: Run agent
    print("\n" + "=" * 60)
    print("STEP 1: Running agent on all CSVs")
    print("=" * 60)
    run_all(test_dir=test_dir)

    # Step 2: Evaluate
    print("\n" + "=" * 60)
    print("STEP 2: Evaluating")
    print("=" * 60)
    run_evaluation(agent_output_dir=AGENT_OUTPUT_DIR)

    # Step 3: Analyze + generate rules
    print("\n" + "=" * 60)
    print("STEP 3: Analyzing traces → generating rules")
    print("=" * 60)
    run_analysis(min_failures=1, agent_output_dir=AGENT_OUTPUT_DIR)

    # Step 4: Validate rules
    print("\n" + "=" * 60)
    print("STEP 4: Validating rules")
    print("=" * 60)
    kept = validate_rules()

    # Summary
    composite_after = compute_composite_for_csvs(
        VALIDATION_CSVS,
        agent_output_dir=AGENT_OUTPUT_DIR,
    )
    rules_after = len(load_rules())

    print("\n" + "=" * 60)
    print("IMPROVEMENT CYCLE COMPLETE")
    print("=" * 60)
    print(f"  Composite: {composite_before:.2%} → {composite_after:.2%} ({composite_after - composite_before:+.2%})")
    print(f"  Rules: {rules_before} → {rules_after} ({'kept' if kept else 'discarded new rules'})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--improve":
            run_improve()
        else:
            run_migration(sys.argv[1])
    else:
        run_all()
