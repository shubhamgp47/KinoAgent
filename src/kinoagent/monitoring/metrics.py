import time
from functools import wraps
from prometheus_client import Counter, Histogram, start_http_server, REGISTRY
from langgraph.errors import GraphInterrupt

def get_or_create_counter(name: str, documentation: str, labelnames=()):
    """Helper to prevent DuplicateTimeseries errors upon module re-import."""
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Counter(name, documentation, labelnames=labelnames)

def get_or_create_histogram(name: str, documentation: str, labelnames=(), buckets=Histogram.DEFAULT_BUCKETS):
    """Helper to prevent DuplicateTimeseries errors upon module re-import."""
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Histogram(name, documentation, labelnames=labelnames, buckets=buckets)

# 1. Metric Definitions
'''A Counter tracks counts of events or running totals.

Example use cases for Counters:

Number of requests processed
Number of items that were inserted into a queue
Total amount of data that a system has processed'''
TOOL_CALL_TOTAL = get_or_create_counter(
    "kinoagent_tool_calls_total",
    "Total invocations of KinoAgent tools",
    labelnames=["tool_name", "status"],
)

'''Histograms sample observations (usually execution duration or request sizes) and place them into predefined numerical bins (buckets).
In Prometheus, _bucket counters allow Grafana to calculate accurate percentiles (e.g., $P_{50}$, $P_{95}$, $P_{99}$) using the histogram_quantile() PromQL function.'''
TOOL_LATENCY_SECONDS = get_or_create_histogram(
    "kinoagent_tool_latency_seconds",
    "Execution duration per tool in seconds",
    labelnames=["tool_name"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 120.0],
)

SQL_GENERATION_FAILURES = get_or_create_counter(
    "kinoagent_sql_generation_failures_total",
    "Total count of rejected or malformed SQL generations",
)

CHAT_NODE_LATENCY = get_or_create_histogram(
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