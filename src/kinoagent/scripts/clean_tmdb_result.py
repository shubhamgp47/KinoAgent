from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_JSONL = (
    BASE_DIR
    / "data"
    / "tmdb_processed_data"
    / "personal_films_enriched.jsonl"
)

OUTPUT_JSONL = (
    BASE_DIR
    / "data"
    / "tmdb_processed_data"
    / "personal_films_compact.jsonl"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    value = value.strip()
    return value if value else None


def top_cast_names(cast: Any, limit: int = 5) -> list[str]:
    if not isinstance(cast, list):
        return []

    names: list[str] = []

    for person in cast:
        if isinstance(person, dict):
            name = clean_string(person.get("name"))
        else:
            name = clean_string(person)

        if name:
            names.append(name)

        if len(names) == limit:
            break

    return names


def string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []

    return [
        value.strip()
        for value in values
        if isinstance(value, str) and value.strip()
    ]


def similar_titles(similar_films: Any) -> list[str]:
    if not isinstance(similar_films, list):
        return []

    titles: list[str] = []
    seen: set[str] = set()

    for film in similar_films:
        if not isinstance(film, dict):
            continue

        title = clean_string(film.get("title"))

        if title and title.casefold() not in seen:
            titles.append(title)
            seen.add(title.casefold())

    return titles


def compact_record(record: dict[str, Any]) -> dict[str, Any] | None:
    tmdb = record.get("tmdb")

    if not isinstance(tmdb, dict):
        return None

    tmdb_id = tmdb.get("tmdb_id")

    if tmdb_id is None:
        return None

    return {
        "film_key": record.get("film_key"),
        "letterboxd_uri": record.get("letterboxd_uri"),
        "tmdb_id": tmdb_id,
        "title": clean_string(tmdb.get("title"))
        or clean_string(record.get("title")),
        "year": record.get("year"),
        "letterboxd_rating": record.get("letterboxd_rating"),
        "overview": clean_string(tmdb.get("overview")),
        "genres": string_list(tmdb.get("genres")),
        "directors": string_list(tmdb.get("directors")),
        "cast": top_cast_names(tmdb.get("cast"), limit=5),
        "runtime_minutes": tmdb.get("runtime_minutes"),
        "tmdb_vote_average": tmdb.get("tmdb_vote_average"),
        "keywords": string_list(tmdb.get("keywords")),
        "similar_titles": similar_titles(tmdb.get("similar_films")),
    }


def main() -> None:
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSONL}")

    enriched_records = read_jsonl(INPUT_JSONL)

    compact_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []

    for record in enriched_records:
        compact = compact_record(record)

        if compact is None:
            skipped_records.append(
                {
                    "film_key": record.get("film_key"),
                    "title": record.get("title"),
                    "year": record.get("year"),
                    "reason": "missing_tmdb_data_or_tmdb_id",
                }
            )
            continue

        compact_records.append(compact)

    write_jsonl(OUTPUT_JSONL, compact_records)

    print(f"Input records: {len(enriched_records)}")
    print(f"Compact records: {len(compact_records)}")
    print(f"Skipped records: {len(skipped_records)}")
    print(f"Created: {OUTPUT_JSONL}")

    if skipped_records:
        skipped_path = OUTPUT_JSONL.with_name(
            "personal_films_compact_skipped.jsonl"
        )
        write_jsonl(skipped_path, skipped_records)
        print(f"Skipped-record report: {skipped_path}")

    if compact_records:
        print("\nFirst compact record:")
        print(
            json.dumps(
                compact_records[0],
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()