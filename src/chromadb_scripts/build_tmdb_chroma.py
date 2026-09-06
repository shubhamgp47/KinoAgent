from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_JSONL = (
    BASE_DIR
    / "tmdb_processed_data"
    / "personal_films_compact.jsonl"
)

CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "tmdb_synopsis"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BATCH_SIZE = 50
TOP_KEYWORDS = 12
TOP_SIMILAR_TITLES = 5


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per line from a JSONL file."""
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

    return records


def is_missing(value: Any) -> bool:
    """Return True for None, NaN, empty strings, and literal 'nan'."""
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    if isinstance(value, str):
        return value.strip().casefold() in {"", "nan", "none", "null"}

    return False


def clean_text(value: Any) -> str:
    """Convert a possibly missing value into clean display text."""
    if is_missing(value):
        return ""

    return " ".join(str(value).split())


def clean_string_list(
    values: Any,
    limit: int | None = None,
) -> list[str]:
    """Clean a list of strings, remove duplicates, and optionally cap it."""
    if not isinstance(values, list):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = clean_text(value)

        if not text:
            continue

        key = text.casefold()

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(text)

        if limit is not None and len(cleaned) >= limit:
            break

    return cleaned


def join_or_unknown(values: list[str]) -> str:
    """Join a cleaned list for document text without inserting empty labels."""
    return ", ".join(values) if values else "Unknown"


def build_tmdb_document(record: dict[str, Any]) -> str:
    """Create the factual text that will be embedded for similarity search."""
    title = clean_text(record.get("title")) or "Unknown title"
    year = clean_text(record.get("year")) or "Unknown year"

    overview = clean_text(record.get("overview"))
    genres = clean_string_list(record.get("genres"))
    directors = clean_string_list(record.get("directors"))
    cast = clean_string_list(record.get("cast"), limit=5)
    keywords = clean_string_list(
        record.get("keywords"),
        limit=TOP_KEYWORDS,
    )
    similar_titles = clean_string_list(
        record.get("similar_titles"),
        limit=TOP_SIMILAR_TITLES,
    )

    sections = [
        f"{title} ({year})",
    ]

    if overview:
        sections.append(f"Overview: {overview}")

    if genres:
        sections.append(f"Genres: {join_or_unknown(genres)}.")

    if directors:
        sections.append(f"Directed by: {join_or_unknown(directors)}.")

    if cast:
        sections.append(f"Starring: {join_or_unknown(cast)}.")

    if keywords:
        sections.append(f"Keywords: {join_or_unknown(keywords)}.")

    if similar_titles:
        sections.append(
            f"Similar films: {join_or_unknown(similar_titles)}."
        )

    return "\n".join(sections)


def build_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Create filterable primitive metadata stored alongside the embedding."""
    metadata: dict[str, Any] = {
        "film_key": clean_text(record.get("film_key")),
        "title": clean_text(record.get("title")),
    }

    tmdb_id = record.get("tmdb_id")
    if not is_missing(tmdb_id):
        metadata["tmdb_id"] = int(tmdb_id)

    year = record.get("year")
    if not is_missing(year):
        metadata["year"] = int(year)

    rating = record.get("letterboxd_rating")
    if not is_missing(rating):
        metadata["letterboxd_rating"] = float(rating)

    runtime = record.get("runtime_minutes")
    if not is_missing(runtime):
        metadata["runtime_minutes"] = int(runtime)

    tmdb_vote_average = record.get("tmdb_vote_average")
    if not is_missing(tmdb_vote_average):
        metadata["tmdb_vote_average"] = float(tmdb_vote_average)

    metadata["genres"] = "|".join(
        clean_string_list(record.get("genres"))
    )

    metadata["directors"] = "|".join(
        clean_string_list(record.get("directors"))
    )

    return {
        key: value
        for key, value in metadata.items()
        if not is_missing(value)
    }


def validate_record(record: dict[str, Any]) -> bool:
    """Accept only records that have a TMDb ID and usable factual text."""
    if is_missing(record.get("tmdb_id")):
        return False

    return bool(
        clean_text(record.get("overview"))
        or clean_string_list(record.get("genres"))
        or clean_string_list(record.get("keywords"))
    )


def batched(
    items: list[dict[str, Any]],
    size: int,
) -> list[list[dict[str, Any]]]:
    """Split records into small batches for stable local ingestion."""
    return [
        items[index:index + size]
        for index in range(0, len(items), size)
    ]

def deduplicate_by_tmdb_id(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep one record per TMDb ID and return duplicates for inspection."""
    unique_records: dict[int, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []

    for record in records:
        tmdb_id = int(record["tmdb_id"])

        if tmdb_id in unique_records:
            duplicates.append(record)
            continue

        unique_records[tmdb_id] = record

    return list(unique_records.values()), duplicates

def main() -> None:
    """Build or refresh the persistent TMDb factual Chroma collection."""
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSONL}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(INPUT_JSONL)

    valid_records = [
    record
    for record in records
    if validate_record(record)
]

    valid_records, duplicate_records = deduplicate_by_tmdb_id(
        valid_records
    )

    skipped_count = len(records) - len(valid_records) - len(duplicate_records)

    print(f"Duplicate TMDb IDs removed: {len(duplicate_records)}")

    if duplicate_records:
        print("Example duplicate:")
        print(
            f"  {duplicate_records[0].get('title')} "
            f"| tmdb_id={duplicate_records[0].get('tmdb_id')} "
            f"| film_key={duplicate_records[0].get('film_key')}"
        )

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={
            "description": (
                "TMDb factual film records for semantic similarity retrieval."
            ),
            "embedding_model": EMBEDDING_MODEL,
        },
    )

    for batch_number, batch in enumerate(
        batched(valid_records, BATCH_SIZE),
        start=1,
    ):
        ids = [
            f"tmdb_{int(record['tmdb_id'])}"
            for record in batch
        ]

        documents = [
            build_tmdb_document(record)
            for record in batch
        ]

        metadatas = [
            build_metadata(record)
            for record in batch
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        print(
            f"Indexed batch {batch_number}: "
            f"{len(batch)} films"
        )

    print("\nFinished indexing TMDb factual records.")
    print(f"Input records: {len(records)}")
    print(f"Indexed records: {len(valid_records)}")
    print(f"Skipped records: {skipped_count}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database folder: {CHROMA_DIR}")
    print(f"Collection count: {collection.count()}")
    print(f"Duplicates removed: {len(duplicate_records)}")

    test_query = (
        "An atmospheric crime thriller about a long investigation "
        "into a serial killer."
    )

    results = collection.query(
        query_texts=[test_query],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    print("\nTest query:")
    print(test_query)

    for index, metadata in enumerate(
        results["metadatas"][0],
        start=1,
    ):
        distance = results["distances"][0][index - 1]

        print(
            f"{index}. {metadata['title']} "
            f"({metadata.get('year', 'Unknown year')}) "
            f"| distance: {distance:.3f}"
        )

if __name__ == "__main__":
    main()