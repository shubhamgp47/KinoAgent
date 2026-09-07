import time
from functools import wraps
from prometheus_client import Counter, Histogram, start_http_server
from langgraph.errors import GraphInterrupt

# 1. Metric Definitions
'''A Counter tracks counts of events or running totals.

Example use cases for Counters:

Number of requests processed
Number of items that were inserted into a queue
Total amount of data that a system has processed'''
TOOL_CALL_TOTAL = Counter(
    "kinoagent_tool_calls_total",
    "Total invocations of KinoAgent tools",
    ["tool_name", "status"],  # status: success | error
)

'''Histograms sample observations (usually execution duration or request sizes) and place them into predefined numerical bins (buckets).
In Prometheus, _bucket counters allow Grafana to calculate accurate percentiles (e.g., $P_{50}$, $P_{95}$, $P_{99}$) using the histogram_quantile() PromQL function.'''
TOOL_LATENCY_SECONDS = Histogram(
    "kinoagent_tool_latency_seconds",
    "Execution duration per tool in seconds",
    ["tool_name"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 120.0],
)

SQL_GENERATION_FAILURES = Counter(
    "kinoagent_sql_generation_failures_total",
    "Total count of rejected or malformed SQL generations",
)

CHAT_NODE_LATENCY = Histogram(
    "kinoagent_chat_node_latency_seconds",
    "Latency of main LLM router node",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# 2. Reusable Metric Decorator
def track_tool_metrics(tool_name: str): # receives the static name label, generating an isolated decorator instance closure.
    """Decorator to measure tool execution duration and outcome."""
    def decorator(func):
        @wraps(func) # @wraps(func): Retains the tool's docstring, function name, and type signatures.
        def wrapper(*args, **kwargs):
            start = time.perf_counter() # Start the timer to measure execution duration.
            try:
                result = func(*args, **kwargs) # Call the actual tool function and store its result.
                # Check for custom error states returned by tools
                if isinstance(result, dict) and result.get("status") == "query_generation_failed":
                    TOOL_CALL_TOTAL.labels(tool_name=tool_name, status="error").inc()
                    SQL_GENERATION_FAILURES.inc()
                else:
                    TOOL_CALL_TOTAL.labels(tool_name=tool_name, status="success").inc()
                return result
            except GraphInterrupt:
                # Do not increment error counters for deliberate HITL pauses
                raise
            except Exception:
                TOOL_CALL_TOTAL.labels(tool_name=tool_name, status="error").inc()
                raise
            finally:
                duration = time.perf_counter() - start # Calculate the total execution time of the tool function.
                TOOL_LATENCY_SECONDS.labels(tool_name=tool_name).observe(duration) # Record the execution duration in the Prometheus histogram for this specific tool.
        return wrapper
    return decorator

# 3. Server Startup
_server_started = False

def start_metrics_endpoint(port: int = 8000):
    """Starts Prometheus metrics endpoint in a background daemon thread."""
    global _server_started
    if not _server_started:
        start_http_server(port)
        _server_started = True