import json
import time
import os
import sys
from pathlib import Path
import sqlite3
import mlflow
from langchain_core.messages import HumanMessage

# Add project root to sys.path so src.kinoagent resolves properly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MLFLOW_DB_PATH = (PROJECT_ROOT / "mlflow.db").resolve()

# Set tracking URI to SQLite database
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
mlflow.set_experiment("KinoAgent_Routing_and_Tool_Evals")

# Import directly from the refactored package
from src.kinoagent.backend import (
    film_gpt,
    generate_oscars_sql,
    is_safe_sql,
    OSCARS_DB_PATH,
)

DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation(model_name: str = "qwen3:8b"):
    # Group runs under a unified MLflow experiment
    mlflow.set_experiment("KinoAgent_Routing_and_Tool_Evals")

    dataset = load_dataset()
    total_tests = len(dataset)
    correct_tool_routes = 0
    valid_sql_generations = 0
    sql_test_count = 0
    total_latency = 0.0

    eval_run_name = f"eval_{model_name}_{int(time.time())}"

    with mlflow.start_run(run_name=eval_run_name):
        # 1. Log static run parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_set_size", total_tests)

        per_test_results = []

        for test in dataset:
            test_id = test["id"]
            query = test["query"]
            expected_tool = test["expected_tool"]
            expected_sql_frag = test.get("expected_sql_fragment")

            start_time = time.perf_counter()

            # Execute in an isolated thread ID so prior test states do not bleed over
            config = {"configurable": {"thread_id": f"eval_thread_{test_id}_{int(time.time())}"}}

            try:
                output = film_gpt.invoke(
                    {"messages": [HumanMessage(content=query)]},
                    config=config,
                )
                duration = time.perf_counter() - start_time
                total_latency += duration

                # Extract tool calls from LangChain message nodes
                messages = output.get("messages", [])
                tools_called = []
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tools_called.extend([tc["name"] for tc in msg.tool_calls])
                    if hasattr(msg, "name") and msg.name:
                        tools_called.append(msg.name)

                tools_called = list(dict.fromkeys(tools_called))
                route_matched = expected_tool in tools_called
                if route_matched:
                    correct_tool_routes += 1

                # Deterministic check for Oscar SQL validity
                sql_status = "N/A"
                if expected_tool == "ask_oscars_database_question":
                    sql_test_count += 1
                    generated_sql = generate_oscars_sql(query)
                    safe = is_safe_sql(generated_sql)
                    
                    syntax_ok = False
                    if safe:
                        try:
                            # Verify read-only query executes cleanly against SQLite
                            db_uri = f"{Path(OSCARS_DB_PATH).resolve().as_uri()}?mode=ro"
                            conn = sqlite3.connect(db_uri, uri=True)
                            cursor = conn.execute(generated_sql)
                            cursor.fetchall()
                            conn.close()
                            syntax_ok = True
                        except Exception:
                            syntax_ok = False

                    fragment_ok = (expected_sql_frag.lower() in generated_sql.lower()) if expected_sql_frag else True

                    if safe and syntax_ok and fragment_ok:
                        valid_sql_generations += 1
                        sql_status = "PASS"
                    else:
                        sql_status = f"FAIL (safe={safe}, syntax={syntax_ok}, frag={fragment_ok})"

                per_test_results.append({
                    "id": test_id,
                    "query": query,
                    "expected_tool": expected_tool,
                    "actual_tools": tools_called,
                    "routing_pass": route_matched,
                    "sql_status": sql_status,
                    "latency_sec": round(duration, 2),
                })

            except Exception as e:
                per_test_results.append({
                    "id": test_id,
                    "query": query,
                    "error": str(e),
                    "routing_pass": False,
                })

        # 2. Compute aggregate metrics
        routing_accuracy = correct_tool_routes / total_tests
        avg_latency = total_latency / total_tests
        sql_accuracy = (valid_sql_generations / sql_test_count) if sql_test_count > 0 else 1.0

        # 3. Log metrics to MLflow tracking store
        mlflow.log_metric("routing_accuracy", routing_accuracy)
        mlflow.log_metric("sql_validity_accuracy", sql_accuracy)
        mlflow.log_metric("avg_latency_seconds", avg_latency)

        # 4. Save test-run details as an artifact
        artifact_file = "eval_results.json"
        with open(artifact_file, "w", encoding="utf-8") as f:
            json.dump(per_test_results, f, indent=2)
        mlflow.log_artifact(artifact_file)
        if os.path.exists(artifact_file):
            os.remove(artifact_file)

        print(f"\n================ Eval Summary ================")
        print(f" Routing Accuracy:     {routing_accuracy:.1%}")
        print(f" SQL Validity:         {sql_accuracy:.1%}")
        print(f" Avg Latency / Query:  {avg_latency:.2f}s")
        print(f"==============================================\n")

if __name__ == "__main__":
    run_evaluation()