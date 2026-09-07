from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv



# Always resolve paths relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_JSONL = BASE_DIR / "data" / "letterboxd_processed_data" / "personal_films.jsonl"

OUTPUT_DIR = BASE_DIR / "data" / "tmdb_processed_data"
OUTPUT_JSONL = OUTPUT_DIR / "personal_films_enriched.jsonl"
MAPPING_CSV = OUTPUT_DIR / "letterboxd_tmdb_mapping.csv"
CACHE_FILE = OUTPUT_DIR / "tmdb_cache.json"
UNMATCHED_FILE = OUTPUT_DIR / "tmdb_unmatched_or_ambiguous.jsonl"


TMDB_BASE_URL = "https://api.themoviedb.org/3"
LANGUAGE = "en-US"

DETAILS_APPEND_FIELDS = (
    "credits,"
    "keywords,"
    "videos,"
    "images,"
    "similar,"
    "recommendations,"
    "release_dates,"
    "watch/providers,"
    "external_ids"
)

TOP_CAST_COUNT = 15
REQUEST_DELAY_SECONDS = 0.20
REQUEST_TIMEOUT_SECONDS = 20


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
                    f"Invalid JSON on line {line_number} in {path}: {error}"
                ) from error

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)


def normalise_title(value: str | None) -> str:
    if not value:
        return ""

    return (
        value.casefold()
        .replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
        .strip()
    )


def release_year(movie: dict[str, Any]) -> int | None:
    release_date = movie.get("release_date") or ""

    if len(release_date) >= 4 and release_date[:4].isdigit():
        return int(release_date[:4])

    return None


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def tmdb_get(
    session: requests.Session,
    api_key: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = session.get(
        f"{TMDB_BASE_URL}{endpoint}",
        headers=build_headers(api_key),
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()
    return response.json()


def score_candidate(
    candidate: dict[str, Any],
    requested_title: str,
    requested_year: int | None,
) -> tuple[int, dict[str, Any]]:
    candidate_title = candidate.get("title") or candidate.get("original_title") or ""
    candidate_year = release_year(candidate)

    title_matches = (
        normalise_title(candidate_title) == normalise_title(requested_title)
    )

    year_matches = (
        requested_year is not None
        and candidate_year is not None
        and candidate_year == requested_year
    )

    year_is_close = (
        requested_year is not None
        and candidate_year is not None
        and abs(candidate_year - requested_year) <= 1
    )

    score = 0

    if title_matches:
        score += 100

    if year_matches:
        score += 50
    elif year_is_close:
        score += 20

    score += min(int(candidate.get("vote_count", 0) / 1000), 10)

    return score, candidate


def choose_tmdb_match(
    results: list[dict[str, Any]],
    title: str,
    year: int | None,
) -> tuple[dict[str, Any] | None, str, list[dict[str, Any]]]:
    if not results:
        return None, "not_found", []

    scored_results = [
        score_candidate(candidate, title, year)
        for candidate in results
    ]

    scored_results.sort(key=lambda item: item[0], reverse=True)

    best_score, best_candidate = scored_results[0]

    candidate_summaries = [
        {
            "tmdb_id": candidate.get("id"),
            "title": candidate.get("title"),
            "release_date": candidate.get("release_date"),
            "vote_count": candidate.get("vote_count"),
            "score": score,
        }
        for score, candidate in scored_results[:5]
    ]

    if best_score >= 150:
        return best_candidate, "matched_exact_title_year", candidate_summaries

    if best_score >= 120:
        return best_candidate, "matched_close_year", candidate_summaries

    return best_candidate, "needs_review", candidate_summaries


def extract_directors(credits: dict[str, Any]) -> list[str]:
    return [
        person["name"]
        for person in credits.get("crew", [])
        if person.get("job") == "Director" and person.get("name")
    ]


def extract_writers(credits: dict[str, Any]) -> list[str]:
    writer_jobs = {"Writer", "Screenplay", "Story", "Characters"}

    return sorted(
        {
            person["name"]
            for person in credits.get("crew", [])
            if person.get("job") in writer_jobs and person.get("name")
        }
    )


def extract_cast(credits: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": person.get("name"),
            "character": person.get("character"),
            "order": person.get("order"),
            "profile_path": person.get("profile_path"),
        }
        for person in credits.get("cast", [])[:TOP_CAST_COUNT]
    ]


def simplify_movie_details(details: dict[str, Any]) -> dict[str, Any]:
    credits = details.get("credits", {})

    return {
        "tmdb_id": details.get("id"),
        "imdb_id": details.get("imdb_id"),
        "title": details.get("title"),
        "original_title": details.get("original_title"),
        "original_language": details.get("original_language"),
        "release_date": details.get("release_date"),
        "status": details.get("status"),
        "overview": details.get("overview"),
        "tagline": details.get("tagline"),
        "runtime_minutes": details.get("runtime"),
        "adult": details.get("adult"),
        "video": details.get("video"),
        "homepage": details.get("homepage"),
        "budget": details.get("budget"),
        "revenue": details.get("revenue"),
        "popularity": details.get("popularity"),
        "tmdb_vote_average": details.get("vote_average"),
        "tmdb_vote_count": details.get("vote_count"),
        "poster_path": details.get("poster_path"),
        "backdrop_path": details.get("backdrop_path"),
        "belongs_to_collection": details.get("belongs_to_collection"),
        "genres": [
            genre.get("name")
            for genre in details.get("genres", [])
            if genre.get("name")
        ],
        "production_companies": [
            company.get("name")
            for company in details.get("production_companies", [])
            if company.get("name")
        ],
        "production_countries": [
            country.get("name")
            for country in details.get("production_countries", [])
            if country.get("name")
        ],
        "spoken_languages": [
            language.get("english_name") or language.get("name")
            for language in details.get("spoken_languages", [])
            if language.get("english_name") or language.get("name")
        ],
        "directors": extract_directors(credits),
        "writers": extract_writers(credits),
        "cast": extract_cast(credits),
        "keywords": [
            keyword.get("name")
            for keyword in details.get("keywords", {}).get("keywords", [])
            if keyword.get("name")
        ],
        "similar_films": [
            {
                "tmdb_id": movie.get("id"),
                "title": movie.get("title"),
                "release_date": movie.get("release_date"),
                "vote_average": movie.get("vote_average"),
            }
            for movie in details.get("similar", {}).get("results", [])[:20]
        ],
        "recommended_films": [
            {
                "tmdb_id": movie.get("id"),
                "title": movie.get("title"),
                "release_date": movie.get("release_date"),
                "vote_average": movie.get("vote_average"),
            }
            for movie in details.get("recommendations", {}).get("results", [])[:20]
        ],
        "videos": [
            {
                "site": video.get("site"),
                "type": video.get("type"),
                "name": video.get("name"),
                "key": video.get("key"),
                "official": video.get("official"),
            }
            for video in details.get("videos", {}).get("results", [])
        ],
        "external_ids": details.get("external_ids", {}),
        "watch_providers": details.get("watch/providers", {}).get("results", {}),
    }


def main() -> None:
    load_dotenv()

    api_key = os.getenv("TMDB_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TMDB_API_KEY is missing from your .env file."
        )

    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSONL}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    personal_films = read_jsonl(INPUT_JSONL)
    cache = load_cache(CACHE_FILE)

    enriched_records: list[dict[str, Any]] = []
    unmatched_records: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []

    session = requests.Session()

    for index, personal_film in enumerate(personal_films, start=1):
        film_key = personal_film["film_key"]
        title = personal_film["title"]
        year = personal_film.get("year")

        print(f"[{index}/{len(personal_films)}] {title} ({year})")

        cached_item = cache.get(film_key)

        if cached_item and cached_item.get("details"):
            details = cached_item["details"]
            match_status = cached_item.get("match_status", "cached")
            candidates = cached_item.get("candidates", [])
        else:
            try:
                search_data = tmdb_get(
                    session=session,
                    api_key=api_key,
                    endpoint="/search/movie",
                    params={
                        "query": title,
                        "year": year,
                        "language": LANGUAGE,
                        "include_adult": "false",
                    },
                )

                match, match_status, candidates = choose_tmdb_match(
                    results=search_data.get("results", []),
                    title=title,
                    year=year,
                )

                if match is None:
                    unmatched_records.append(
                        {
                            **personal_film,
                            "tmdb_match_status": "not_found",
                            "tmdb_candidates": [],
                        }
                    )

                    mapping_rows.append(
                        {
                            "film_key": film_key,
                            "title": title,
                            "year": year,
                            "tmdb_id": None,
                            "tmdb_title": None,
                            "tmdb_release_date": None,
                            "match_status": "not_found",
                        }
                    )

                    time.sleep(REQUEST_DELAY_SECONDS)
                    continue

                tmdb_id = match["id"]

                details = tmdb_get(
                    session=session,
                    api_key=api_key,
                    endpoint=f"/movie/{tmdb_id}",
                    params={
                        "language": LANGUAGE,
                        "append_to_response": DETAILS_APPEND_FIELDS,
                    },
                )

                cache[film_key] = {
                    "tmdb_id": tmdb_id,
                    "match_status": match_status,
                    "candidates": candidates,
                    "details": details,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }

                save_cache(CACHE_FILE, cache)
                time.sleep(REQUEST_DELAY_SECONDS)

            except requests.RequestException as error:
                print(f"  ERROR: {error}")

                unmatched_records.append(
                    {
                        **personal_film,
                        "tmdb_match_status": "request_error",
                        "tmdb_error": str(error),
                    }
                )
                continue

        tmdb_data = simplify_movie_details(details)

        enriched_record = {
            **personal_film,
            "tmdb_match_status": match_status,
            "tmdb_candidates": candidates,
            "tmdb": tmdb_data,
        }

        enriched_records.append(enriched_record)

        mapping_rows.append(
            {
                "film_key": film_key,
                "title": title,
                "year": year,
                "tmdb_id": tmdb_data["tmdb_id"],
                "tmdb_title": tmdb_data["title"],
                "tmdb_release_date": tmdb_data["release_date"],
                "match_status": match_status,
            }
        )

        if match_status == "needs_review":
            unmatched_records.append(enriched_record)

    write_jsonl(OUTPUT_JSONL, enriched_records)
    write_jsonl(UNMATCHED_FILE, unmatched_records)

    import pandas as pd

    pd.DataFrame(mapping_rows).to_csv(MAPPING_CSV, index=False)

    print("\nFinished.")
    print(f"Enriched films: {len(enriched_records)}")
    print(f"Needs review / unmatched: {len(unmatched_records)}")
    print(f"Enriched output: {OUTPUT_JSONL}")
    print(f"TMDb mapping: {MAPPING_CSV}")
    print(f"Review file: {UNMATCHED_FILE}")
    print(f"Cache: {CACHE_FILE}")


if __name__ == "__main__":
    main()