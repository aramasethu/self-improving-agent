"""
Self-Improving Data Migration Agent
====================================
Inspired by Remote's use case: takes messy CSV employee data and maps it
to a standardized HR schema using an LLM agent built with LangGraph.

Workshop: Agent Engineering with LangGraph + LangSmith
"""

import json
import os
import sys
import uuid
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ---------------------------------------------------------------------------
# Target schema - every CSV must be mapped to this
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
# LLM setup - Claude Haiku
# ---------------------------------------------------------------------------
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    max_tokens=2048,
)

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    csv_path: str
    raw_columns: list[str]
    sample_rows: list[dict]
    column_mapping: dict  # {source_col: target_col}
    generated_code: str
    transformed_data: list[dict]
    validation_result: dict
    error: str


# ---------------------------------------------------------------------------
# Node 1: Analyze the input CSV
# ---------------------------------------------------------------------------
def analyze_csv(state: AgentState) -> AgentState:
    """Read the CSV and extract column names + sample rows."""
    csv_path = state["csv_path"]
    df = pd.read_csv(csv_path)

    return {
        "raw_columns": df.columns.tolist(),
        "sample_rows": df.head(3).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Node 2: Map columns using LLM
# ---------------------------------------------------------------------------
def map_columns(state: AgentState) -> AgentState:
    """Ask the LLM to map source columns to target schema columns."""

    target_desc = json.dumps(TARGET_SCHEMA, indent=2)
    source_cols = json.dumps(state["raw_columns"])
    sample = json.dumps(state["sample_rows"], indent=2, default=str)

    response = llm.invoke([
        SystemMessage(content=f"""You are a data migration expert. Your job is to map source CSV columns to a target schema.

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


# ---------------------------------------------------------------------------
# Node 3: Generate Pandas transformation code
# ---------------------------------------------------------------------------
def generate_code(state: AgentState) -> AgentState:
    """Generate Python/Pandas code to transform the data."""

    mapping = json.dumps(state["column_mapping"], indent=2)
    sample = json.dumps(state["sample_rows"], indent=2, default=str)
    target = json.dumps(TARGET_SCHEMA, indent=2)

    response = llm.invoke([
        SystemMessage(content=f"""You are a Python/Pandas expert. Generate code to transform a CSV dataframe.

The code must:
1. Assume `df` is already loaded as a pandas DataFrame
2. Rename columns according to the mapping
3. Convert the start_date column to YYYY-MM-DD format (handle various date formats)
4. Ensure salary_usd is a float
5. Select only the target schema columns
6. Assign the result to a variable called `result_df`

Target schema:
{target}

IMPORTANT: Return ONLY the Python code, no markdown fences, no explanation."""),
        HumanMessage(content=f"""Column mapping (source -> target): {mapping}

Sample data:
{sample}

Generate the transformation code."""),
    ])

    code = response.content.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()
    return {"generated_code": code}


# ---------------------------------------------------------------------------
# Node 4: Execute the generated code
# ---------------------------------------------------------------------------
def execute_code(state: AgentState) -> AgentState:
    """Execute the generated Pandas code on the actual CSV."""
    try:
        df = pd.read_csv(state["csv_path"])
        exec_globals = {"pd": pd, "re": __import__("re"), "datetime": __import__("datetime")}
        local_vars = {"df": df}
        exec(state["generated_code"], exec_globals, local_vars)
        result_df = local_vars["result_df"]
        return {"transformed_data": result_df.to_dict(orient="records")}
    except Exception as e:
        return {"error": f"Code execution failed: {e}"}


# ---------------------------------------------------------------------------
# Node 5: Validate the output
# ---------------------------------------------------------------------------
def validate_output(state: AgentState) -> AgentState:
    """Check the transformed data against the target schema."""
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

    return {
        "validation_result": {
            "valid": len(issues) == 0,
            "issues": issues,
            "row_count": len(data),
        }
    }


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze_csv", analyze_csv)
    graph.add_node("map_columns", map_columns)
    graph.add_node("generate_code", generate_code)
    graph.add_node("execute_code", execute_code)
    graph.add_node("validate_output", validate_output)

    graph.add_edge(START, "analyze_csv")
    graph.add_edge("analyze_csv", "map_columns")
    graph.add_edge("map_columns", "generate_code")
    graph.add_edge("generate_code", "execute_code")
    graph.add_edge("execute_code", "validate_output")
    graph.add_edge("validate_output", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
AGENT_OUTPUT_DIR = "results/agent_outputs"


def run_migration(csv_path: str, run_id: uuid.UUID | None = None, metadata: dict | None = None):
    """Run the data migration agent on a CSV file and save results to disk."""
    print(f"Processing: {csv_path}")

    app = build_graph()

    if run_id is None:
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
    }

    run_metadata = {"csv_file": os.path.basename(csv_path)}
    if metadata:
        run_metadata.update(metadata)

    config = {
        "run_id": run_id,
        "run_name": f"migrate-{os.path.basename(csv_path)}",
        "metadata": run_metadata,
        "tags": metadata.get("tags", []) if metadata else [],
    }
    result = app.invoke(initial_state, config=config)
    result["_run_id"] = str(run_id)

    # Save result to disk
    os.makedirs(AGENT_OUTPUT_DIR, exist_ok=True)
    # Name output after the input CSV: test_000.csv -> test_000.json
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
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"  Saved: {output_path}")
    return result


def run_all(test_dir: str = "data/test", ground_truth_path: str = "data/ground_truth.json"):
    """Run the agent on all test CSVs and save each result."""
    # Load ground truth for metadata tagging
    gt_lookup = {}
    if os.path.exists(ground_truth_path):
        with open(ground_truth_path) as f:
            for entry in json.load(f):
                gt_lookup[entry["csv_file"]] = entry

    csv_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".csv"))
    print(f"Running agent on {len(csv_files)} CSVs from {test_dir}/\n")

    for csv_file in csv_files:
        csv_path = os.path.join(test_dir, csv_file)

        # Build metadata from ground truth
        gt_entry = gt_lookup.get(csv_file, {})
        metadata = {
            "challenges": gt_entry.get("challenges", []),
            "date_format": gt_entry.get("date_format", ""),
            "salary_format": gt_entry.get("salary_format", ""),
            "tags": gt_entry.get("challenges", []),
        }

        try:
            run_migration(csv_path, metadata=metadata)
        except Exception as e:
            print(f"  ERROR: {e}")
            os.makedirs(AGENT_OUTPUT_DIR, exist_ok=True)
            base_name = os.path.splitext(csv_file)[0]
            output_path = os.path.join(AGENT_OUTPUT_DIR, f"{base_name}.json")
            with open(output_path, "w") as f:
                json.dump({
                    "csv_file": csv_file,
                    "csv_path": csv_path,
                    "raw_columns": [],
                    "sample_rows": [],
                    "column_mapping": {},
                    "generated_code": "",
                    "transformed_data": [],
                    "validation_result": {},
                    "error": str(e),
                    "run_id": "",
                }, f, indent=2)

    print(f"\nDone. Results saved to {AGENT_OUTPUT_DIR}/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_migration(sys.argv[1])
    else:
        run_all()
