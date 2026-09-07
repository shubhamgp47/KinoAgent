import mlflow
import json
import time
from backend_watchlist_tool import film_gpt
from langchain_core.messages import HumanMessage

mlflow.set_experiment("FilmGPT_Model_Evaluation")

# Enable automatic tracing for LangChain / LangGraph components
mlflow.langchain.autolog()

EVAL_BENCHMARK = [
    {
        "query": "How many Oscars has Leonardo DiCaprio won?",
        "expected_tool": "ask_oscars_database_question",
    },
    {
        "query": "Find movies similar to Zodiac with a dark mood.",
        "expected_tool": "synopsis_retriever",
    }
]

def run_evaluation(model_name: str):
    with mlflow.start_run(run_name=f"eval_{model_name}_{int(time.time())}"):
        mlflow.log_param("model_name", model_name)
        
        correct_routings = 0
        total_latency = 0.0

        for item in EVAL_BENCHMARK:
            start = time.perf_counter()
            config = {"configurable": {"thread_id": f"eval_{time.time()}"}}
            
            output = film_gpt.invoke(
                {"messages": [HumanMessage(content=item["query"])]},
                config=config,
            )
            elapsed = time.perf_counter() - start
            total_latency += elapsed

            # Validate whether the expected tool message exists in execution steps
            messages = output.get("messages", [])
            tools_used = [
                msg.name for msg in messages if hasattr(msg, "name") and msg.name
            ]
            
            if item["expected_tool"] in tools_used:
                correct_routings += 1

        accuracy = correct_routings / len(EVAL_BENCHMARK)
        avg_latency = total_latency / len(EVAL_BENCHMARK)

        # Log system-level ML metrics
        mlflow.log_metric("routing_accuracy", accuracy)
        mlflow.log_metric("avg_turn_latency_seconds", avg_latency)
        
        print(f"Eval completed: Accuracy={accuracy:.2%}, Avg Latency={avg_latency:.2f}s")

if __name__ == "__main__":
    run_evaluation("gemini-2.5-flash")