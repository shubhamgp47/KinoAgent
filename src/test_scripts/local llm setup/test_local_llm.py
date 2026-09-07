"""
test_local_llm.py
------------------
Pulls small Qwen3 models via Ollama and tests tool-calling compatibility
for the FilmGPT-style LangGraph agent, on an RTX 3060 Laptop (6GB VRAM).

Fix applied: resolves the full path to ollama.exe explicitly, since conda
environments on Windows sometimes don't inherit the user PATH that a
regular terminal has.

Usage:
    python test_local_llm.py
"""

import os
import shutil
import subprocess
import sys
import time

MODELS_TO_TEST = ["qwen3:4b", "qwen3:8b"]  # 4b = safe fit, 8b = tight fit on 6GB


def find_ollama_exe() -> str:
    """
    Locate ollama.exe robustly:
    1. Try shutil.which (respects current PATH).
    2. Fall back to the standard Windows install location.
    3. Raise a clear error if neither works.
    """
    found = shutil.which("ollama")
    if found:
        return found

    common_paths = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
        r"C:\Program Files\Ollama\ollama.exe",
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        "Could not locate ollama.exe. Checked PATH and common install "
        "locations. Open a normal terminal (not this conda env) and run "
        "'where ollama' to find it, then hardcode OLLAMA_EXE below."
    )


OLLAMA_EXE = find_ollama_exe()
print(f"Using ollama executable: {OLLAMA_EXE}")


def run_cmd(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, shell=False)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")


def pull_models():
    for model in MODELS_TO_TEST:
        print(f"\n=== Pulling {model} ===")
        run_cmd([OLLAMA_EXE, "pull", model])


def check_ollama_running():
    try:
        import ollama
        ollama.list()
        return True
    except Exception as e:
        print(f"Ollama server not reachable: {e}")
        print("Start the Ollama app from the Start menu, then retry.")
        return False


# ---------------------------------------------------------------------------
# Tool-calling test — mirrors your FilmGPT setup structurally:
# a Tavily-like web search tool + a local retriever-like tool.
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    """
    Get the current weather for a city.

    Args:
        city: Name of the city to check weather for.

    Returns:
        A short weather description.
    """
    return f"It's 18C and partly cloudy in {city}."


def search_films(query: str) -> str:
    """
    Search for films matching a description (stands in for synopsis_retriever).

    Args:
        query: Natural language description of the film to search for.

    Returns:
        A short list of matching film titles.
    """
    return f"Top matches for '{query}': Blade Runner 2049, Arrival, Dune."


def test_model_basic_chat(model: str):
    from ollama import chat

    print(f"\n--- Basic chat test: {model} ---")
    t0 = time.time()
    response = chat(
        model=model,
        messages=[{"role": "user", "content": "In one sentence, what is a film noir?"}],
    )
    elapsed = time.time() - t0
    print(f"Response ({elapsed:.1f}s): {response.message.content}")


def test_model_tool_calling(model: str):
    from ollama import chat

    print(f"\n--- Tool-calling test: {model} ---")
    messages = [
        {
            "role": "user",
            "content": "What's the weather in Berlin, and can you find me a sci-fi film similar to Blade Runner?",
        }
    ]

    t0 = time.time()
    response = chat(
        model=model,
        messages=messages,
        tools=[get_weather, search_films],
    )
    elapsed = time.time() - t0

    print(f"Time: {elapsed:.1f}s")
    print(f"Content: {response.message.content!r}")

    if response.message.tool_calls:
        print(f"Tool calls detected: {len(response.message.tool_calls)}")
        for tc in response.message.tool_calls:
            print(f"  -> {tc.function.name}({tc.function.arguments})")
    else:
        print("No tool calls detected — model answered directly (check if this is expected).")


def test_langchain_integration(model: str):
    """
    Mirrors your actual backend.py usage: ChatOllama + bind_tools.
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.tools import tool
        from langchain_core.messages import HumanMessage
    except ImportError:
        print("langchain-ollama not installed. Run: pip install langchain-ollama")
        return

    @tool
    def weather_tool(city: str) -> str:
        """Get current weather for a city."""
        return f"It's 18C and partly cloudy in {city}."

    @tool
    def film_search_tool(query: str) -> str:
        """Search for films matching a description."""
        return f"Top matches for '{query}': Blade Runner 2049, Arrival, Dune."

    print(f"\n--- LangChain bind_tools test: {model} ---")
    llm = ChatOllama(model=model, temperature=0.3)
    llm_with_tools = llm.bind_tools([weather_tool, film_search_tool])

    t0 = time.time()
    response = llm_with_tools.invoke(
        [HumanMessage(content="What's the weather in Berlin, and find me a sci-fi film like Blade Runner?")]
    )
    elapsed = time.time() - t0

    print(f"Time: {elapsed:.1f}s")
    print(f"Content: {response.content!r}")
    print(f"Tool calls: {response.tool_calls}")


def main():
    if not check_ollama_running():
        sys.exit(1)

    pull_models()

    for model in MODELS_TO_TEST:
        print(f"\n{'='*60}\nTESTING MODEL: {model}\n{'='*60}")
        try:
            test_model_basic_chat(model)
            test_model_tool_calling(model)
            test_langchain_integration(model)
        except Exception as e:
            print(f"Error testing {model}: {e}")

    print("\n\nDone. Compare timing + tool-call accuracy above to pick your model for backend.py.")


if __name__ == "__main__":
    main()