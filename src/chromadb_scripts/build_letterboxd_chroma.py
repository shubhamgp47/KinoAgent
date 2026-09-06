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
    / "data" 
    / "letterboxd_processed_data"
    / "personal_films.jsonl"
)

CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "letterboxd_personal"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BATCH_SIZE = 50


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records, where each non-empty line is one film."""
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
    """Identify null-like values without embedding words such as 'NaN'."""
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    if isinstance(value, str):
        return value.strip().casefold() in {"", "nan", "none", "null"}

    return False


def clean_text(value: Any) -> str:
    """Return normalized text or an empty string for missing values."""
    if is_missing(value):
        return ""

    return " ".join(str(value).split())


def clean_dates(value: Any) -> list[str]:
    """Keep valid watch-date strings and discard missing values."""
    if not isinstance(value, list):
        return []

    return [
        clean_text(item)
        for item in value
        if clean_text(item)
    ]


def build_personal_document(record: dict[str, Any]) -> str:
    """Build subjective Letterboxd text for personal-taste retrieval."""
    title = clean_text(record.get("title")) or "Unknown title"
    year = clean_text(record.get("year")) or "Unknown year"

    rating = record.get("my_letterboxd_rating")
    if is_missing(rating):
        rating_text = "My rating: not rated."
    else:
        rating_text = f"My rating: {float(rating):.1f} stars."

    watch_dates = clean_dates(record.get("watch_dates"))
    if watch_dates:
        watched_text = f"Watch dates: {', '.join(watch_dates)}."
    else:
        watched_text = "Watched: yes. No diary date recorded."

    review = clean_text(record.get("review_text"))
    if review:
        review_text = f"My review: {review}"
    else:
        review_text = "I did not write a review."

    return "\n".join(
        [
            f"{title} ({year})",
            rating_text,
            watched_text,
            review_text,
        ]
    )


def build_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Store simple fields for filtering, debugging, and later joins."""
    metadata: dict[str, Any] = {
        "film_key": clean_text(record.get("film_key")),
        "letterboxd_uri": clean_text(record.get("letterboxd_uri")),
        "title": clean_text(record.get("title")),
        "watched": bool(record.get("watched", True)),
    }

    year = record.get("year")
    if not is_missing(year):
        metadata["year"] = int(year)

    rating = record.get("my_letterboxd_rating")
    if not is_missing(rating):
        metadata["letterboxd_rating"] = float(rating)

    return {
        key: value
        for key, value in metadata.items()
        if not is_missing(value)
    }


def valid_record(record: dict[str, Any]) -> bool:
    """Require a stable local identifier and a film title."""
    return bool(
        clean_text(record.get("film_key"))
        and clean_text(record.get("title"))
    )


def batched(
    items: list[dict[str, Any]],
    size: int,
) -> list[list[dict[str, Any]]]:
    """Split work into manageable batches for Chroma upserts."""
    return [
        items[index:index + size]
        for index in range(0, len(items), size)
    ]


def main() -> None:
    """Build or refresh the personal Letterboxd Chroma collection."""
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSONL}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(INPUT_JSONL)

    records = [
        record
        for record in records
        if valid_record(record)
    ]

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={
            "description": (
                "Personal Letterboxd ratings, watch dates, and reviews."
            ),
            "embedding_model": EMBEDDING_MODEL,
        },
    )

    for batch_number, batch in enumerate(
        batched(records, BATCH_SIZE),
        start=1,
    ):
        ids = [
            f"letterboxd_{record['film_key']}"
            for record in batch
        ]

        documents = [
            build_personal_document(record)
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
            f"{len(batch)} Letterboxd films"
        )

    print("\nFinished indexing personal Letterboxd records.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Indexed records: {len(records)}")
    print(f"Collection count: {collection.count()}")
    print(f"Database folder: {CHROMA_DIR}")

    test_query = (
        "Films I enjoyed that explore obsessive investigations "
        "or dark psychological crime."
    )

    results = collection.query(
        query_texts=[test_query],
        n_results=5,
        include=["metadatas", "distances"],
    )

    print("\nTest query:")
    print(test_query)

    for index, metadata in enumerate(
        results["metadatas"][0],
        start=1,
    ):
        print(
            f"{index}. {metadata['title']} "
            f"({metadata.get('year', 'Unknown year')}) "
            f"| Your rating: "
            f"{metadata.get('letterboxd_rating', 'not rated')} "
            f"| distance: {results['distances'][0][index - 1]:.3f}"
        )


if __name__ == "__main__":
    main()