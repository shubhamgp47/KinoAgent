import json
import time
import os
import sqlite3
import mlflow
from langchain_core.messages import HumanMessage

# Import the compiled LangGraph agent and SQL generator from your backend
from kinoagent.backend import film_gpt, generate_oscars_sql, is_safe_sql, OSCARS_DB_PATH

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation(model_name: str = "qwen3:8b"):
    # Set the experiment name in MLflow
    mlflow.set_experiment("KinoAgent_Routing_and_Tool_Evals")

    dataset = load_dataset()
    total_tests = len(dataset)
    correct_tool_routes = 0
    valid_sql_generations = 0
    sql_test_count = 0
    total_latency = 0.0

    eval_run_name = f"eval_{model_name}_{int(time.time())}"

    print(f"\n=======================================================")
    print(f" Starting Automated Evaluation Run: {eval_run_name}")
    print(f" Test Cases: {total_tests} | Model: {model_name}")
    print(f"=======================================================\n")

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

            print(f"Executing [{test_id}]: \"{query}\"")
            start_time = time.perf_counter()

            # Execute graph turn in an isolated ephemeral thread
            config = {"configurable": {"thread_id": f"eval_thread_{test_id}_{int(time.time())}"}}
            
            try:
                output = film_gpt.invoke(
                    {"messages": [HumanMessage(content=query)]},
                    config=config,
                )
                duration = time.perf_counter() - start_time
                total_latency += duration

                # Extract tool calls invoked by the agent
                messages = output.get("messages", [])
                tools_called = []
                for msg in messages:
                    # Capture tool calls attached to AI messages
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tools_called.extend([tc["name"] for tc in msg.tool_calls])
                    # Also capture ToolMessages returned by tool executions
                    if hasattr(msg, "name") and msg.name:
                        tools_called.append(msg.name)

                # Deduplicate tools called in this turn
                tools_called = list(dict.fromkeys(tools_called))

                # Check routing correctness
                route_matched = expected_tool in tools_called
                if route_matched:
                    correct_tool_routes += 1

                # Check SQL correctness if this was an Oscar query
                sql_status = "N/A"
                if expected_tool == "ask_oscars_database_question":
                    sql_test_count += 1
                    generated_sql = generate_oscars_sql(query)
                    safe = is_safe_sql(generated_sql)
                    
                    # Verify query executes safely against SQLite without throwing syntax errors
                    syntax_ok = False
                    if safe:
                        try:
                            conn = sqlite3.connect(f"file:{OSCARS_DB_PATH}?mode=ro", uri=True)
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

                result_record = {
                    "id": test_id,
                    "query": query,
                    "expected_tool": expected_tool,
                    "actual_tools": tools_called,
                    "routing_pass": route_matched,
                    "sql_status": sql_status,
                    "latency_sec": round(duration, 2),
                }
                per_test_results.append(result_record)
                print(f"   -> Result: Route={'PASS' if route_matched else 'FAIL'} | Latency={duration:.2f}s\n")

            except Exception as e:
                print(f"   -> ERROR on {test_id}: {e}\n")
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

        # 3. Log metrics to MLflow
        mlflow.log_metric("routing_accuracy", routing_accuracy)
        mlflow.log_metric("sql_validity_accuracy", sql_accuracy)
        mlflow.log_metric("avg_latency_seconds", avg_latency)

        # 4. Save test-run details as an artifact
        artifact_path = "eval_results.json"
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(per_test_results, f, indent=2)
        mlflow.log_artifact(artifact_path)
        if os.path.exists(artifact_path):
            os.remove(artifact_path)

        print(f"=======================================================")
        print(f" Evaluation Completed Successfully")
        print(f" Routing Accuracy:     {routing_accuracy:.1%}")
        print(f" SQL Accuracy:         {sql_accuracy:.1%}")
        print(f" Avg Latency / Query:  {avg_latency:.2f}s")
        print(f"=======================================================\n")

if __name__ == "__main__":
    run_evaluation()