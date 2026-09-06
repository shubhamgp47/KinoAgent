from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

import os
from typing import Any, List

import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

# Main chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

# ---------------------------------------------------------------------------
# ChromaDB setup: persistent client + embedding model + helpers
# ---------------------------------------------------------------------------

# Base data paths. Adjust if your project layout differs.
#DATA_ROOT = os.getenv("FILMGPT_DATA_ROOT", "data")
CHROMA_PATH = "C:\\Users\\shubh\\OneDrive\\Documents\\Tutorials\\FilmGPT\\FilmGPT\\data\\chroma_db"

# Create a single persistent Chroma client for the whole backend.
# This points at the folder with chroma.sqlite3 and HNSW index folders.
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Use the same local sentence-transformer embedding function used
# when building the collections. This must match the ingestion scripts.
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def get_tmdb_collection():
    """
    Helper to load the TMDb synopsis collection with an attached embedding
    function for semantic retrieval.

    Collection name: "tmdb_synopsis"
    Purpose: factual/semantic questions about plots, genres, cast, directors,
    themes, and "films similar to X".
    """
    return chroma_client.get_collection(
        name="tmdb_synopsis",
        embedding_function=embedding_fn,
    )

def get_letterboxd_collection():
    """
    Helper to load the personal Letterboxd collection with an attached
    embedding function for semantic retrieval.

    Collection name: "letterboxd_personal"
    Purpose: subjective questions about *your* viewing history, ratings,
    reviews, and taste.
    """
    return chroma_client.get_collection(
        name="letterboxd_personal",
        embedding_function=embedding_fn,
    )

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# Web search fallback tool for current/external information
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
)


@tool
def personaltaste_retriever(query: str) -> str:
    """
    Retrieve information about YOUR personal film taste, viewing history,
    ratings and reviews from the 'letterboxd_personal' Chroma collection.

    Use this tool when the user asks about:
    - Films they have watched, rated, or reviewed.
    - Patterns in their own ratings and opinions (e.g. favourites, least liked).
    - Their past viewing history for specific directors, actors, or themes.

    Important:
    - This tool only uses the user's Letterboxd data (titles, ratings, diary
      entries, and review text). It does NOT directly access TMDb genres or
      keywords.
    - For detailed genre, plot, and cast information, call 'synopsis_retriever'
      in addition to this tool.

    Args:
        query: Natural-language description of what to search for in my
               personal Letterboxd data.

    Returns:
        A formatted text summary of the top matching films, including title,
        year, my rating, and a snippet of any review text if available.
    """
    collection = get_letterboxd_collection()

    result = collection.query(
        query_texts=[query],
        n_results=5,
    )

    # Unwrap the outer list-of-lists returned by Chroma
    ids_batches = result.get("ids") or []
    metadatas_batches = result.get("metadatas") or []
    documents_batches = result.get("documents") or []

    if not ids_batches or not metadatas_batches or not documents_batches:
        return "I couldn't find any matching personal Letterboxd records for that query."

    ids = ids_batches[0]
    metadatas = metadatas_batches[0]
    documents = documents_batches[0]

    if not metadatas or not documents:
        return "I couldn't find any matching personal Letterboxd records for that query."

    lines = []
    for idx, (md, doc) in enumerate(zip(metadatas, documents), start=1):
        # md is now a dict, so .get() is valid
        title = md.get("title", "Unknown title")
        year = md.get("year", "Unknown year")
        my_rating = md.get("letterboxd_rating", md.get("letterboxdrating", "N/A"))
        watched = md.get("watched", True)

        doc_snippet = doc[:600]
        if len(doc) > 600:
            doc_snippet += "..."

        lines.append(
            f"Result {idx}\n"
            f"Title: {title} ({year})\n"
            f"Watched: {watched}\n"
            f"My rating: {my_rating}\n"
            f"Personal notes:\n{doc_snippet}\n"
        )

    return "\n\n".join(lines)


@tool
def synopsis_retriever(query: str) -> str:
    """
    Retrieve TMDb-based factual and semantic information from the 'tmdb_synopsis'
    Chroma collection.

    Use cases can be:
    - Film plots, synopses, genres, moods, and themes.
    - Directors, key cast members, runtime, and TMDb ratings.
    - Finding films similar to a description, mood, or another film.

    Important:
    - This tool uses TMDb-enriched documents (overview, genres, directors,
      cast, keywords, similar titles) and associated metadata.
    - It does NOT directly know whether the user personally liked a film;
      for personal ratings and reviews, use 'personaltaste_retriever'.

    Args:
        query: Natural-language description of the desired film plots,
               genres, moods, themes, or cast/director features.

    Returns:
        A formatted text summary of top matching films including title,
        year, genres, directors, key cast, and a synopsis snippet.
    """
    collection = get_tmdb_collection()

    result = collection.query(
        query_texts=[query],
        n_results=5,
    )

    ids_batches = result.get("ids") or []
    metadatas_batches = result.get("metadatas") or []
    documents_batches = result.get("documents") or []

    if not ids_batches or not metadatas_batches or not documents_batches:
        return "I couldn't find any matching TMDb synopsis records for that query."

    ids = ids_batches[0]
    metadatas = metadatas_batches[0]
    documents = documents_batches[0]

    lines: List[str] = []
    for idx, (md, doc) in enumerate(zip(metadatas, documents), start=1):
        title = md.get("title", "Unknown title")
        year = md.get("year", "Unknown year")
        genres = md.get("genres", "")
        directors = md.get("directors", "")
        tmdb_vote_avg = md.get("tmdb_vote_average", md.get("tmdbvoteaverage", "N/A"))

        doc_snippet = doc[:700]
        if len(doc) > 700:
            doc_snippet += "..."

        lines.append(
            f"Result {idx}\n"
            f"Title: {title} ({year})\n"
            f"Genres: {genres}\n"
            f"Directors: {directors}\n"
            f"TMDb average rating: {tmdb_vote_avg}\n"
            f"Synopsis:\n{doc_snippet}\n"
        )

    return "\n\n".join(lines)


# You can add histogram and Oscars tools later; for now we focus on the
# two Chroma retrievers plus web search for current/external info.

# Tool list bound to the LLM.
tools = [
    search_tool,
    personaltaste_retriever,
    synopsis_retriever,
]

llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------------------------
# LangGraph state, nodes, and graph compilation
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    """
    Conversation state shared across the graph.

    'messages' is a list of LangChain messages; add_messages ensures
    new messages are appended automatically as nodes run.
    """
    messages: Annotated[List[BaseMessage], add_messages]


def chat_node(state: ChatState) -> ChatState:
    """
    Main LLM node.

    It decides whether to answer directly or call one of the tools:
    - personaltaste_retriever for questions about YOUR ratings, reviews, and history.
    - synopsis_retriever for TMDb-based plots, genres, cast, directors, and similarity.
    - search_tool only for current/external info outside your local datasets.
    """
    system_message = SystemMessage(
    content=(
        "You are FilmGPT, a personal film assistant that routes questions to "
        "specialized tools over the user's local film data.\n\n"
        "Tools and routing rules:\n"
        "- Use 'personaltaste_retriever' for any question about the user's own "
        "Letterboxd data: watched films, the user's ratings, diary dates, and "
        "personal reviews. This tool is for subjective taste and viewing history.\n"
        "- Use 'synopsis_retriever' for semantic questions about film plots, "
        "genres, moods, themes, directors, cast, and for finding films similar "
        "to a description or title based on TMDb synopses.\n"
        "- Use 'search_tool' only for current or external information that is "
        "not present in the local film data, such as up-to-date streaming "
        "availability, very recent releases, or news.\n\n"
        "Important limitations:\n"
        "- 'personaltaste_retriever' does NOT directly access TMDb genres or "
        "keywords; it only sees the user's titles, ratings, diary entries, and "
        "review text.\n"
        "- 'synopsis_retriever' does NOT know whether the user personally liked "
        "a film; it only provides factual TMDb information.\n\n"
        "Multi-tool guidance:\n"
        "- For questions like \"What kind of films do I usually enjoy?\" first "
        "use 'personaltaste_retriever' to identify films the user rated highly, "
        "then use 'synopsis_retriever' to analyze their genres and themes.\n"
        "- For questions like \"Which dark psychological films have I rated "
        "highly?\" start with 'personaltaste_retriever' and, if needed, call "
        "'synopsis_retriever' to confirm the genres or mood.\n\n"
        "General guidelines:\n"
        "- Prefer local tools over web search when the answer can be derived "
        "from the user's Letterboxd or TMDb enrichment data.\n"
        "- When a tool returns results, read them carefully, reason about them, "
        "and then respond in a clear, conversational way.\n"
        "- If a query mixes personal taste and factual film information, you "
        "may call more than one tool in sequence.\n"
        "- If the user asks general opinions or explanations that do not "
        "require external data, you can answer directly without tools.\n"
    ))

    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": state["messages"] + [response]}


# Single ToolNode that executes whichever tool the LLM requested.
tool_node = ToolNode(tools)

# Sqlite-based checkpointing (thread persistence)
conn = sqlite3.connect("film_gpt.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

# Build the graph with the same START -> chat -> tools_condition -> tools -> chat loop.
graph = StateGraph(ChatState)

# Register nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Edges: start at chat, then conditionally go to tools, then back to chat
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# Compile the graph with checkpointing enabled.
film_gpt = graph.compile(checkpointer=checkpoint)


def get_all_threads() -> list[str]:
    """
    Helper for the Streamlit frontend: list all existing thread IDs
    stored in the SqliteSaver checkpoint.
    """
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config["configurable"]["thread_id"])
    return list(all_threads)