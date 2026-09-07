from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
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
import re

from monitoring.metrics import (
    start_metrics_endpoint,
    track_tool_metrics,
    CHAT_NODE_LATENCY,
)

# Local llm
from langchain_ollama import ChatOllama

import os
from pathlib import Path
from typing import Any, List

import chromadb
from chromadb.utils import embedding_functions

import requests
from langgraph.types import interrupt

load_dotenv()

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

# Main chat model

'''llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.7,
)'''

'''llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)'''

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.7,
)

# local llm
'''llm = ChatOllama(
    model="qwen3:4b",
    temperature=0.7,
)'''

# ---------------------------------------------------------------------------
# ChromaDB setup: persistent client + embedding model + helpers
# ---------------------------------------------------------------------------

# Base data paths. Adjust if your project layout differs.
#DATA_ROOT = os.getenv("FILMGPT_DATA_ROOT", "data")
#CHROMA_PATH = "C:\\Users\\shubh\\OneDrive\\Documents\\Tutorials\\FilmGPT\\FilmGPT\\data\\chroma_db"
#OSCARS_DB_PATH = "C:\\Users\\shubh\\OneDrive\\Documents\\Tutorials\\FilmGPT\\FilmGPT\\data\\oscars_db\\oscars.db"
TMDB_V3_API_KEY = os.getenv("TMDB_V3_API_KEY")
TMDB_SESSION_ID = os.getenv("TMDB_SESSION_ID")
TMDB_ACCOUNT_ID = os.getenv("TMDB_ACCOUNT_ID")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
DATA_DIR = Path(os.getenv("KINOAGENT_DATA_DIR", PROJECT_ROOT / "data")).resolve()
CHROMA_PATH = str(DATA_DIR / "chroma_db")
OSCARS_DB_PATH = (DATA_DIR / "oscars_db" / "oscars.db").resolve()
CHECKPOINT_DB_PATH = str(DATA_DIR / "kinoagent_state.db")

# MLOps Telemetry Server
# Starts a daemon thread serving metrics on http://localhost:8000
start_metrics_endpoint(port=8000)

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

def tmdb_search_movie(title: str) -> dict | None:
    """
    Search TMDb for a movie title and return the best (first) match,
    including its numeric TMDb id, release year, and poster info.
    Returns None if nothing is found.
    """
    resp = requests.get(
        f"{TMDB_BASE_URL}/search/movie",
        params={"api_key": TMDB_V3_API_KEY, "query": title},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return None

    top = results[0]
    return {
        "tmdb_id": top["id"],
        "title": top.get("title", title),
        "year": (top.get("release_date") or "????")[:4],
    }

'''def tmdb_add_to_watchlist(tmdb_id: int) -> dict:
    """
    Perform the actual write: add a movie to the authenticated account's
    TMDb watchlist. Requires TMDB_SESSION_ID and TMDB_ACCOUNT_ID to be set.
    """
    resp = requests.post(
        f"{TMDB_BASE_URL}/account/{TMDB_ACCOUNT_ID}/watchlist",
        params={"api_key": TMDB_V3_API_KEY, "session_id": TMDB_SESSION_ID},
        json={
            "media_type": "movie",
            "media_id": tmdb_id,
            "watchlist": True,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()'''


def clean_film_title(raw_title: str) -> str:
    """
    Remove trailing year annotations like '(1997)' or '- 1997' that the
    LLM sometimes copies from prior conversation context, since TMDb's
    search matches better against a bare title.
    """
    cleaned = re.sub(r"\s*[\(\[]\s*\d{4}\s*[\)\]]\s*$", "", raw_title)
    cleaned = re.sub(r"\s*-\s*\d{4}\s*$", "", cleaned)
    return cleaned.strip()

def tmdb_set_watchlist_status(tmdb_id: int, on_watchlist: bool) -> dict:
    """
    Perform the actual write: add or remove a movie from the authenticated
    account's TMDb watchlist. Requires TMDB_SESSION_ID and TMDB_ACCOUNT_ID.

    TMDb uses the SAME endpoint for add and remove — the only difference
    is the boolean 'watchlist' field in the request body.
    """
    resp = requests.post(
        f"{TMDB_BASE_URL}/account/{TMDB_ACCOUNT_ID}/watchlist",
        params={"api_key": TMDB_V3_API_KEY, "session_id": TMDB_SESSION_ID},
        json={
            "media_type": "movie",
            "media_id": tmdb_id,
            "watchlist": on_watchlist,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()

def tmdb_get_watchlist_movies() -> list[dict]:
    """
    Fetch the full list of movies currently on the account's TMDb watchlist,
    across all pages. Returns a list of dicts with id, title, and year.
    """
    all_movies = []
    page = 1

    while True:
        resp = requests.get(
            f"{TMDB_BASE_URL}/account/{TMDB_ACCOUNT_ID}/watchlist/movies",
            params={
                "api_key": TMDB_V3_API_KEY,
                "session_id": TMDB_SESSION_ID,
                "page": page,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("results", []):
            all_movies.append({
                "tmdb_id": item["id"],
                "title": item.get("title", "Unknown title"),
                "year": (item.get("release_date") or "????")[:4],
            })

        total_pages = data.get("total_pages", 1)
        if page >= total_pages:
            break
        page += 1

    return all_movies

def tmdb_get_movie_details(tmdb_id: int) -> dict:
    """
    Fetch full details for a movie by its TMDb id, including credits
    (cast/director) in a single call via append_to_response.
    """
    resp = requests.get(
        f"{TMDB_BASE_URL}/movie/{tmdb_id}",
        params={
            "api_key": TMDB_V3_API_KEY,
            "append_to_response": "credits",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()

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
@track_tool_metrics("personaltaste_retriever")
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
        n_results=15,
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
@track_tool_metrics("synopsis_retriever")
def synopsis_retriever(query: str) -> str:
    """
    Retrieve TMDb-based factual and semantic information from the 'tmdb_synopsis'
    Chroma collection, INCLUDING the user's own Letterboxd rating for each film
    if one exists.

    Use cases can be:
    - Film plots, synopses, genres, moods, and themes.
    - Directors, key cast members, runtime, and TMDb ratings.
    - Finding films similar to a description, mood, or another film.
    - Questions that combine mood/genre/theme AND the user's own rating,
      e.g. "dark psychological films I've rated highly" — this tool alone
      can answer these since it carries both genre/mood data AND the
      user's letterboxd_rating in the same record.

    Important:
    - This tool uses TMDb-enriched documents (overview, genres, directors,
      cast, keywords, similar titles) and associated metadata, PLUS the
      user's own letterboxd_rating where available.
    - If letterboxd_rating is missing or "N/A" for a film, the user has
      likely not rated it (or it isn't in their watched history).

    Args:
        query: Natural-language description of the desired film plots,
            genres, moods, themes, or cast/director features.

    Returns:
        A formatted text summary of top matching films including title,
        year, genres, directors, TMDb rating, the user's own rating
        (if available), and a synopsis snippet.
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

        # NEW LINE: read the user's own rating if present in this
        # collection's metadata. Same fallback pattern as personaltaste_retriever.
        my_rating = md.get("letterboxd_rating", md.get("letterboxdrating", "N/A"))

        doc_snippet = doc[:700]
        if len(doc) > 700:
            doc_snippet += "..."

        lines.append(
            f"Result {idx}\n"
            f"Title: {title} ({year})\n"
            f"Genres: {genres}\n"
            f"Directors: {directors}\n"
            f"TMDb average rating: {tmdb_vote_avg}\n"
            f"My rating: {my_rating}\n"   # NEW LINE in the output
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
@track_tool_metrics("ask_oscars_database_question")
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
    db_file = Path(OSCARS_DB_PATH).resolve()
    if not db_file.is_file():
        raise FileNotFoundError(f"Oscars database file missing at: {db_file}")
    #db_uri = f"file:{OSCARS_DB_PATH}?mode=ro"
    db_uri = f"{db_file.as_uri()}?mode=ro"
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

@tool
@track_tool_metrics("update_tmdb_watchlist")
def update_tmdb_watchlist(
    film_title: str,
    action: Literal["add", "remove"],
) -> str:
    """
    Add or remove a film from the user's TMDb account watchlist.

    This is a WRITE action and requires human approval before it executes,
    regardless of direction (add or remove).

    Use this tool whenever the user asks to:
    - Add a film to their watchlist, save it for later, or want to watch it
      later -> call with action="add".
    - Remove a film from their watchlist, take it off the list, or say they
      no longer want to watch it -> call with action="remove".

    Do NOT use this tool for questions about films the user has already
    watched, rated, or reviewed — that is what 'personaltaste_retriever'
    is for. This tool only modifies the TMDb watchlist itself.

    Args:
        film_title: The title of the film, as mentioned by the user.
        action: Either "add" or "remove".

    Returns:
        A confirmation message if approved and successfully applied, or a
        message stating the action was rejected or the film could not be found.
    """
    search_title = clean_film_title(film_title)
    match = tmdb_search_movie(search_title)

    if match is None:
        # Retry once with the raw title in case cleaning over-stripped something
        match = tmdb_search_movie(film_title)

    if match is None:
        return f"I couldn't find a TMDb entry for '{film_title}'. Please check the title."

    on_watchlist = (action == "add")
    verb = "Add" if on_watchlist else "Remove"
    verb_past = "added to" if on_watchlist else "removed from"
    verb_present = "add" if on_watchlist else "remove"

    # Pause the graph here. The payload is what your Streamlit sidebar
    # will render inside pending_hitl["prompt"].
    decision = interrupt({
        "action": f"{action}_tmdb_watchlist",
        "film_title": match["title"],
        "year": match["year"],
        "tmdb_id": match["tmdb_id"],
        "message": (
            f"{verb} '{match['title']}' ({match['year']}) "
            f"{'to' if on_watchlist else 'from'} your TMDb watchlist?"
        ),
    })

    if decision != "yes":
        return (
            f"Okay, I did not {verb_present} '{match['title']}' "
            f"({match['year']}) {'to' if on_watchlist else 'from'} your watchlist."
        )

    try:
        tmdb_set_watchlist_status(match["tmdb_id"], on_watchlist)
    except requests.HTTPError as error:
        return f"The TMDb API rejected the request: {error}"

    return f"'{match['title']}' ({match['year']}) has been {verb_past} your TMDb watchlist."

@tool
@track_tool_metrics("get_watchlist_summary")
def get_watchlist_summary(list_titles: bool = False) -> str:
    """
    Get the number of films on the user's TMDb watchlist, and optionally
    list their titles.

    Use this tool when the user asks things like:
    - "How many films are in my watchlist?"
    - "What's on my watchlist?"
    - "List my watchlist."

    Args:
        list_titles: If True, include the full list of film titles and
            years in the response. If False, return only the count.

    Returns:
        A summary string with the watchlist count, and optionally the
        list of films.
    """
    movies = tmdb_get_watchlist_movies()
    count = len(movies)

    if not list_titles or count == 0:
        return f"There are {count} film(s) on your TMDb watchlist."

    listing = "\n".join(
        f"- {m['title']} ({m['year']})" for m in movies
    )
    return f"There are {count} film(s) on your TMDb watchlist:\n{listing}"

@tool
@track_tool_metrics("tmdb_movie_lookup")
def tmdb_movie_lookup(film_title: str) -> str:
    """
    Look up ANY film on TMDb's full catalog and return its details:
    genres, director, cast, rating, release year, and synopsis.

    Use this tool when the user asks about a film that is NOT necessarily
    in their personal Letterboxd collection or local TMDb synopsis
    embeddings — for example films they have not watched or rated yet.
    This tool queries TMDb's live catalog directly, so it can find any
    film TMDb has indexed, not just films already in the local database.

    Prefer 'synopsis_retriever' first ONLY if the question is about a film
    likely already in the user's local collection (e.g. they mention having
    watched, rated, or reviewed it). If 'synopsis_retriever' returns no match,
    or the user is clearly asking about an unfamiliar/unwatched film, use
    this tool instead of guessing or apologizing.

    Args:
        film_title: The title of the film to look up.

    Returns:
        A formatted text summary of the film's genres, director, cast,
        TMDb rating, release year, and synopsis, or a not-found message.
    """
    search_title = clean_film_title(film_title)
    match = tmdb_search_movie(search_title)

    if match is None:
        match = tmdb_search_movie(film_title)

    if match is None:
        return f"I couldn't find '{film_title}' on TMDb. Please check the title."

    details = tmdb_get_movie_details(match["tmdb_id"])

    genres = ", ".join(g["name"] for g in details.get("genres", []))

    crew = details.get("credits", {}).get("crew", [])
    directors = ", ".join(
        c["name"] for c in crew if c.get("job") == "Director"
    ) or "Unknown"

    cast_list = details.get("credits", {}).get("cast", [])
    top_cast = ", ".join(c["name"] for c in cast_list[:5]) or "Unknown"

    rating = details.get("vote_average", "N/A")
    year = (details.get("release_date") or "????")[:4]
    overview = details.get("overview", "No synopsis available.")

    return (
        f"Title: {details.get('title', match['title'])} ({year})\n"
        f"Genres: {genres}\n"
        f"Director: {directors}\n"
        f"Starring: {top_cast}\n"
        f"TMDb Rating: {rating}\n"
        f"Synopsis: {overview}"
    )

# You can add histogram and Oscars tools later; for now we focus on the
# two Chroma retrievers plus web search for current/external info.

# Tool list bound to the LLM.
tools = [
    search_tool,
    personaltaste_retriever,
    synopsis_retriever,
    ask_oscars_database_question,
    update_tmdb_watchlist,
    get_watchlist_summary,
    tmdb_movie_lookup,
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
        "Response format rules (very important):\n"
        "- NEVER narrate your reasoning, tool selection process, or planning out "
        "loud. Do not write phrases like 'Since the data is already retrieved' or "
        "'The assistant should respond with'. Go straight to the final answer.\n"
        "- NEVER use placeholder brackets like [Title 1] or [Rating]. Always "
        "substitute the actual values returned by the tool.\n"
        "- Your response must consist ONLY of the direct answer to the user, "
        "written in plain conversational language.\n\n"
        "- Use 'personaltaste_retriever' for any question about the user's own "
        "Letterboxd data: watched films, the user's ratings, diary dates, and "
        "personal reviews. This tool is for subjective taste and viewing history.\n"
        "- For questions combining mood, genre, or theme WITH the user's own "
        "rating (e.g. \"dark psychological films I've rated highly\", \"what "
        "comedies did I love\"), use 'synopsis_retriever' alone — it contains "
        "both TMDb genre/mood data and the user's own rating in the same record. "
        "Only use 'personaltaste_retriever' for pure taste/history questions that "
        "don't need genre or plot context, e.g. \"what have I watched recently\" "
        "or \"show me my reviews\".\n\n"
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
        "- Use 'update_tmdb_watchlist' when the user asks to add a film to their "
        "watchlist, save it for later, or wants to watch it in the future "
        "(action=\"add\"), OR when the user asks to remove a film from their "
        "watchlist, take it off the list, or no longer wants to watch it "
        "(action=\"remove\"). This is a write action that requires the user's "
        "explicit approval, which happens automatically outside of your control — "
        "just call the tool with the correct action and report back its result "
        "message to the user verbatim.\n\n"
        "- Use 'tmdb_movie_lookup' for ANY film the user asks about that might not "
        "be in the local synopsis_retriever collection — especially films they say "
        "they haven't watched, or any film where 'synopsis_retriever' returns no "
        "match or an unclear match. This tool queries TMDb's full live catalog, so "
        "it can answer questions about literally any film TMDb has indexed. Never "
        "tell the user a well-known film 'isn't in the database' without trying "
        "this tool first.\n"
        "- Use 'get_watchlist_summary' when the user asks how many films are on "
        "their watchlist, or asks to see what's on it. Set list_titles=True only "
        "if they ask to see the actual titles, not just a count.\n\n"
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
    #response = llm_with_tools.invoke(messages)

    with CHAT_NODE_LATENCY.time():
        response = llm_with_tools.invoke(messages)

    #return {"messages": state["messages"] + [response]}
    return {"messages": [response]}


# Single ToolNode that executes whichever tool the LLM requested.
tool_node = ToolNode(tools)

# Sqlite-based checkpointing (thread persistence)
#conn = sqlite3.connect("film_gpt.db", check_same_thread=False)
#CHECKPOINT_DB_PATH = str(DATA_DIR / "kinoagent_state.db")
#conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
checkpoint_conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
checkpoint = SqliteSaver(checkpoint_conn)
#checkpoint = SqliteSaver(conn)

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