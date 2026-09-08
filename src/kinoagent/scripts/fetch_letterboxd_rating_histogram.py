from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


# Resolve project root no matter where the script is executed from
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_JSONL = BASE_DIR / "data" / "letterboxd_processed_data" / "personal_films.jsonl"

OUTPUT_DIR = BASE_DIR / "data" / "letterboxd_processed_data"
OUTPUT_JSONL = OUTPUT_DIR / "rating_histograms.jsonl"


REQUEST_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 20
MAX_FILMS = None  # Start with 10. Set to None only after testing.

RATING_VALUES = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
]

USER_AGENT = (
    "FilmGPT personal learning project "
    "(contact: replace-with-your-email@example.com)"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


'''def letterboxd_slug(uri: str | None) -> str | None:
    if not uri:
        return None

    match = re.search(r"boxd\.it/([A-Za-z0-9]+)", uri)

    if not match:
        return None

    return match.group(1)'''


def resolve_letterboxd_slug(
    session: requests.Session,
    short_uri: str | None,
) -> str | None:
    if not short_uri:
        return None

    response = session.get(
        short_uri,
        headers={"User-Agent": USER_AGENT},
        allow_redirects=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    path_parts = [
        part
        for part in urlparse(response.url).path.split("/")
        if part
    ]

    if len(path_parts) >= 2 and path_parts[0] == "film":
        return path_parts[1]

    return None


def parse_count(value: str) -> int:
    value = value.strip().replace(",", "")

    match = re.search(r"([\d.]+)\s*([KM]?)", value, flags=re.IGNORECASE)

    if not match:
        raise ValueError(f"Could not parse rating count: {value}")

    number = float(match.group(1))
    suffix = match.group(2).upper()

    multiplier = {
        "": 1,
        "K": 1_000,
        "M": 1_000_000,
    }[suffix]

    return int(number * multiplier)


def parse_histogram(html: str) -> tuple[dict[str, int], float, int]:
    soup = BeautifulSoup(html, "html.parser")

    distribution: dict[str, int] = {}

    rating_lookup = {
        "%C2%BD": 0.5,
        "1": 1.0,
        "1%C2%BD": 1.5,
        "2": 2.0,
        "2%C2%BD": 2.5,
        "3": 3.0,
        "3%C2%BD": 3.5,
        "4": 4.0,
        "4%C2%BD": 4.5,
        "5": 5.0,
    }

    for link in soup.select('a[href*="/ratings/rated/"]'):
        href = link.get("href", "")

        rating_match = re.search(
            r"/ratings/rated/([^/]+)/by/rating",
            href,
            flags=re.IGNORECASE,
        )

        if not rating_match:
            continue

        encoded_rating = rating_match.group(1).upper()
        rating_value = rating_lookup.get(encoded_rating)

        if rating_value is None:
            continue

        tooltip = (
            link.get("data-original-title")
            or link.get("title")
            or ""
        ).replace("\xa0", " ")

        count_match = re.search(
            r"([\d,.]+(?:[KM])?)\s+.*?ratings?",
            tooltip,
            flags=re.IGNORECASE,
        )

        if not count_match:
            continue

        distribution[str(rating_value)] = parse_count(
            count_match.group(1)
        )

    missing_buckets = [
        str(rating)
        for rating in RATING_VALUES
        if str(rating) not in distribution
    ]

    if missing_buckets:
        raise ValueError(
            "Histogram parsing failed. Missing buckets: "
            f"{missing_buckets}. Parsed: {distribution}"
        )

    average_link = soup.select_one(
        'a[data-original-title*="Weighted average"]'
    )

    tooltip = ""
    if average_link is not None:
        tooltip = average_link.get(
            "data-original-title",
            "",
        )

    if not tooltip:
        average_link = soup.find(
            "a",
            title=re.compile(
                r"Weighted average of",
                flags=re.IGNORECASE,
            ),
        )

        if average_link is not None:
            tooltip = average_link.get("title", "")

    average_match = re.search(
        r"Weighted average of\s+(\d+(?:\.\d+)?)\s+"
        r"based on\s+([\d,]+)\s+ratings",
        tooltip,
        flags=re.IGNORECASE,
    )

    if not average_match:
        meta_average = soup.select_one(
            'meta[name="twitter:data2"]'
        )

        meta_content = (
            meta_average.get("content", "")
            if meta_average is not None
            else ""
        )

        meta_match = re.search(
            r"(\d+(?:\.\d+)?)\s+out of\s+5",
            meta_content,
            flags=re.IGNORECASE,
        )

        if not meta_match:
            raise ValueError(
                "Could not find Letterboxd weighted average. "
                f"Average tooltip found: {tooltip!r}"
            )

        community_weighted_average = float(meta_match.group(1))
        total_ratings = sum(distribution.values())

    else:
        community_weighted_average = float(
            average_match.group(1)
        )
        total_ratings = int(
            average_match.group(2).replace(",", "")
        )

    return distribution, community_weighted_average, total_ratings


def calculate_comparison(
    my_rating: float | None,
    distribution: dict[str, int],
    weighted_average: float,
) -> dict[str, float | None]:
    if my_rating is None:
        return {
            "rating_delta_from_average": None,
            "share_ratings_below_mine": None,
            "share_ratings_at_or_below_mine": None,
        }

    total = sum(distribution.values())

    below = sum(
        count
        for rating, count in distribution.items()
        if float(rating) < my_rating
    )

    at_or_below = sum(
        count
        for rating, count in distribution.items()
        if float(rating) <= my_rating
    )

    return {
        "rating_delta_from_average": round(my_rating - weighted_average, 2),
        "share_ratings_below_mine": round(below / total, 4),
        "share_ratings_at_or_below_mine": round(
            at_or_below / total,
            4,
        ),
    }


def fetch_histogram(
    session: requests.Session,
    slug: str,
) -> tuple[dict[str, int], float, int]:
    url = (
        "https://letterboxd.com/csi/film/"
        f"{slug}/rating-histogram/"
    )

    response = session.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    return parse_histogram(response.text)


def main() -> None:
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(
            f"Could not find input file: {INPUT_JSONL}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    personal_films = read_jsonl(INPUT_JSONL)

    existing_records = (
        read_jsonl(OUTPUT_JSONL)
        if OUTPUT_JSONL.exists()
        else []
    )

    existing_keys = {
        record["film_key"]
        for record in existing_records
    }

    session = requests.Session()
    new_records = []

    candidates = [
        film
        for film in personal_films
        if film["film_key"] not in existing_keys
        and film.get("letterboxd_rating") is not None
    ]

    if MAX_FILMS is not None:
        candidates = candidates[:MAX_FILMS]

    for index, film in enumerate(candidates, start=1):
        film_key = film["film_key"]
        slug = resolve_letterboxd_slug(
        session=session,
        short_uri=film.get("letterboxd_uri"),
    )

        print(
            f"[{index}/{len(candidates)}] "
            f"{film['title']} ({film['year']})"
        )

        if not slug:
            print("  SKIPPED: Could not derive a Letterboxd slug.")
            continue

        try:
            distribution, average, total = fetch_histogram(
                session=session,
                slug=slug,
            )

            my_rating = float(film["letterboxd_rating"])
            comparison = calculate_comparison(
                my_rating=my_rating,
                distribution=distribution,
                weighted_average=average,
            )

            new_records.append(
                {
                    "film_key": film_key,
                    "title": film["title"],
                    "year": film["year"],
                    "letterboxd_uri": film.get("letterboxd_uri"),
                    "histogram_url": (
                        "https://letterboxd.com/csi/film/"
                        f"{slug}/rating-histogram/"
                    ),
                    "histogram_fetched_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "my_rating": my_rating,

                    # Exact value extracted from Letterboxd's "Weighted average of ..."
                    # tooltip. Keep this for accurate calculations.
                    "community_weighted_average": average,

                    # One-decimal version that matches the rating shown on Letterboxd's UI.
                    "community_display_rating": round(average, 1),

                    "community_rating_count": total,
                    "rating_distribution": distribution,
                    **comparison,
                }
            )

            community_display_rating = round(average, 1)

            print(
                f"  Letterboxd display rating: {community_display_rating} | "
                f"Exact weighted rating: {average:.2f} | "
                f"Your rating: {my_rating} | "
                f"Delta: {comparison['rating_delta_from_average']:+.2f}"
            )

        except requests.HTTPError as error:
            print(f"  HTTP ERROR: {error}")

        except (ValueError, KeyError) as error:
            print(f"  PARSE ERROR: {error}")

        time.sleep(REQUEST_DELAY_SECONDS)

    all_records = existing_records + new_records
    write_jsonl(OUTPUT_JSONL, all_records)

    print("\nFinished.")
    print(f"New records: {len(new_records)}")
    print(f"Total cached records: {len(all_records)}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()