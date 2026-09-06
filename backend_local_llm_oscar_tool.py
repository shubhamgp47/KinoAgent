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

# Local llm
from langchain_ollama import ChatOllama

import os
from typing import Any, List

import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

# Main chat model
'''llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)'''

# local llm
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0.7,
)

# ---------------------------------------------------------------------------
# ChromaDB setup: persistent client + embedding model + helpers
# ---------------------------------------------------------------------------

# Base data paths. Adjust if your project layout differs.
#DATA_ROOT = os.getenv("FILMGPT_DATA_ROOT", "data")
CHROMA_PATH = "C:\\Users\\shubh\\OneDrive\\Documents\\Tutorials\\FilmGPT\\FilmGPT\\data\\chroma_db"
OSCARS_DB_PATH = "C:\\Users\\shubh\\OneDrive\\Documents\\Tutorials\\FilmGPT\\FilmGPT\\data\\oscars_db\\oscars.db"

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



# A small, focused LLM just for SQL generation
sql_llm = ChatOllama(
    model="qwen3:4b",
    temperature=0.0,
)

'''sql_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
)'''

OSCARS_SCHEMA_PROMPT = """
Return exactly one valid SQLite SELECT statement and nothing else.

Use this exact database table name:
oscars_nominations

Do not use any other table name, including:
nominations, nominations_table, oscar_nominations, academy_awards.

Available columns in oscars_nominations:
year_film, year_ceremony, ceremony, category, canon_category,
nominee_name, film, winner

Definitions:
- Every row is one Oscar nomination.
- A row with winner = 1 is both a nomination and a win.
- Total nominations: COUNT(*).
- Total wins: SUM(CASE WHEN winner = 1 THEN 1 ELSE 0 END).
- Non-winning nominations: SUM(CASE WHEN winner = 0 THEN 1 ELSE 0 END).
- Number of categories: COUNT(DISTINCT canon_category).

Rules:
- Use only SELECT.
- Use COLLATE NOCASE when matching film or nominee_name.
- Use LIMIT 50 only for a query that lists multiple rows.
- Never use explanations, Markdown, comments, placeholders, or code fences.
- Never use a semicolon.

Required query patterns:
- "How many times has PERSON been nominated?"
  SELECT COUNT(*) AS total_nominations
  FROM oscars_nominations
  WHERE nominee_name COLLATE NOCASE = 'PERSON'

- "How many Oscars did FILM win?"
  SELECT SUM(CASE WHEN winner = 1 THEN 1 ELSE 0 END) AS total_wins
  FROM oscars_nominations
  WHERE film COLLATE NOCASE = 'FILM'

- "Did FILM win any Oscars?"
  SELECT SUM(CASE WHEN winner = 1 THEN 1 ELSE 0 END) AS total_wins
  FROM oscars_nominations
  WHERE film COLLATE NOCASE = 'FILM'

User question:
{question}
"""

# To generate sql select query to answer a user question about Oscars nominations/wins.
def generate_oscars_sql(question: str) -> str:
    prompt = OSCARS_SCHEMA_PROMPT.format(question=question)
    response = sql_llm.invoke(prompt)

    raw_output = str(response.content).strip()

    # Remove accidental Markdown fences.
    raw_output = raw_output.replace("```sql", "")
    raw_output = raw_output.replace("```SQL", "")
    raw_output = raw_output.replace("```", "").strip()

    # Discard any accidental prose before SELECT.
    select_index = raw_output.upper().find("SELECT")
    if select_index == -1:
        return raw_output

    sql = raw_output[select_index:].strip()

    # Retain one statement only and remove its optional terminal semicolon.
    if ";" in sql:
        sql = sql.split(";", 1).strip()

    return sql

def is_safe_sql(sql: str) -> bool:
    sql_upper = sql.upper()
    if not sql_upper.startswith("SELECT"):
        return False
    if "OSCARS_NOMINATIONS" not in sql_upper:
        return False
    forbidden = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
        "ATTACH", "DETACH", "PRAGMA", "CREATE", "REPLACE", "VACUUM",
    ]
    if any(word in sql_upper for word in forbidden):
        return False
    if ";" in sql:
        return False
    return True

@tool
def ask_oscars_database_question(question: str) -> dict:
    """
    Answer questions about Oscar nominations and wins using a local SQLite database.

    Examples:
    - "Did Zodiac win any Oscars?"
    - "Was Toy Story 3 nominated for an Oscar?"
    - "How many Oscars has Leonardo DiCaprio won?"
    - "Who won Best Director in 1995?"

    The tool returns:
    - 'rows': a list of dicts (each dict is one row of the result)
    - 'row_count': number of rows
    """
    sql = generate_oscars_sql(question)

    if not is_safe_sql(sql):
        return {
            "status": "query_generation_failed",
            "error": "Generated SQL was unsafe or invalid.",
            "rows": [],
            "row_count": 0,
        }                               

    db_uri = f"file:{OSCARS_DB_PATH}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    try:
        cursor = conn.execute(sql)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
    finally:
        conn.close()

    result_rows = [
        dict(zip(columns, row))
        for row in rows
    ]

    return {
        "rows": result_rows,
        "row_count": len(result_rows),
    }
# You can add histogram and Oscars tools later; for now we focus on the
# two Chroma retrievers plus web search for current/external info.

# Tool list bound to the LLM.
tools = [
    search_tool,
    personaltaste_retriever,
    synopsis_retriever,
    ask_oscars_database_question,
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
        "- Use 'ask_oscars_database_question' for any Oscar nomination, win, "
        "category, ceremony, actor/director awards question covered by the "
        "local oscars.db SQLite database. Do NOT guess Oscar results; always "
        "query the database via this tool.\n"
        "Base your answer ONLY on the structured 'rows' and 'row_count' returned "
        "by the tool. Do NOT show raw SQL queries, internal schema details, or "
        "intermediate reasoning to the user. Summarize the result in one or two "
        "plain-language sentences.\n"
        "- Use 'search_tool' only for current or external information that is "
        "not present in the local film data, such as up-to-date streaming "
        "availability, very recent releases, or news.\n\n"
        "Important limitations:\n"
        "- 'personaltaste_retriever' does NOT directly access TMDb genres or "
        "keywords; it only sees the user's titles, ratings, diary entries, and "
        "review text.\n"
        "- 'synopsis_retriever' does NOT know whether the user personally liked "
        "a film; it only provides factual TMDb information.\n"
        "- 'ask_oscars_database_question' is the authoritative source for Oscar "
        "questions. If the database returns no rows, explain that the film or "
        "person has no entries in the local dataset instead of inventing awards.\n\n"
        "- If 'ask_oscars_database_question' returns an error or "
        "'status' equal to 'query_generation_failed', do not interpret the "
        "empty rows as zero nominations or zero wins. Instead, say that the "
        "local Oscars database query could not be completed and ask the user "
        "to try again.\n\n"
        "Multi-tool guidance:\n"
        "- For questions like \"What kind of films do I usually enjoy?\" first "
        "use 'personaltaste_retriever' to identify films the user rated highly, "
        "then use 'synopsis_retriever' to analyze their genres and themes.\n"
        "- For questions like \"Which dark psychological films have I rated "
        "highly?\" start with 'personaltaste_retriever' and, if needed, call "
        "'synopsis_retriever' to confirm the genres or mood.\n"
        "- For questions like \"Did Zodiac win any Oscars?\" or \"How many Oscars "
        "has Leonardo DiCaprio won?\" always call 'ask_oscars_database_question' "
        "and base your answer on its rows.\n\n"
        "General guidelines:\n"
        "- Prefer local tools over web search when the answer can be derived "
        "from the user's Letterboxd, TMDb enrichment data, or oscars.db.\n"
        "- When a tool returns results, read them carefully, reason about them, "
        "and then respond in a clear, conversational way.\n"
        "- If a query mixes personal taste and factual film information, you "
        "may call more than one tool in sequence.\n"
        "- If the user asks general opinions or explanations that do not "
        "require external data, you can answer directly without tools.\n"
    )
)

    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": state["messages"] + [response]}
    #return {"messages": [response]}


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