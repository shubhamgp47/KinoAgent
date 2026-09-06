from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "letterboxd_raw_data"
PROCESSED_DIR = BASE_DIR / "data" / "letterboxd_processed_data"

WATCHED_FILE = RAW_DIR/"watched.csv"
RATINGS_FILE = RAW_DIR / "ratings.csv"
DIARY_FILE = RAW_DIR / "diary.csv"
REVIEWS_FILE = RAW_DIR / "reviews.csv"


OUTPUT_CSV = PROCESSED_DIR / "personal_films.csv"
OUTPUT_JSONL = PROCESSED_DIR / "personal_films.jsonl"


def clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None

    value = str(value).strip()
    return value if value else None


def normalise_uri(value: object) -> str | None:
    uri = clean_text(value)

    if uri is None:
        return None

    if uri.startswith("[") and "](" in uri and uri.endswith(")"):
        return uri.split("](", maxsplit=1)[1][:-1]

    return uri


def normalise_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)

    return text


def film_key(row: pd.Series) -> str:
    title = normalise_text(row.get("Name"))
    year = normalise_text(row.get("Year"))

    return f"{title}:{year}"

'''def film_key(row: pd.Series) -> str:
    uri = normalise_uri(row.get("Letterboxd URI"))

    if uri:
        return uri

    title = clean_text(row.get("Name")) or ""
    year = clean_text(row.get("Year")) or ""
    return f"{title.casefold()}::{year}"'''


def read_export(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required export: {path}")

    dataframe = pd.read_csv(path)

    dataframe["film_key"] = dataframe.apply(film_key, axis=1)
    dataframe["letterboxd_uri"] = dataframe["Letterboxd URI"].map(normalise_uri)

    return dataframe


def unique_non_empty(values: pd.Series) -> list[str]:
    cleaned = [clean_text(value) for value in values]
    return sorted({value for value in cleaned if value is not None})


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    watched = read_export(WATCHED_FILE)
    ratings = read_export(RATINGS_FILE)
    diary = read_export(DIARY_FILE)
    reviews = read_export(REVIEWS_FILE)

    print("\nRaw column names:")
    print("watched:", watched.columns.tolist())
    print("diary:", diary.columns.tolist())
    print("reviews:", reviews.columns.tolist())

    print("\nSample keys:")
    print("watched:", watched[["Name", "Year", "Letterboxd URI", "film_key"]].head(3).to_dict("records"))
    print("diary:", diary[["Name", "Year", "Letterboxd URI", "film_key"]].head(3).to_dict("records"))
    print("reviews:", reviews[["Name", "Year", "Letterboxd URI", "film_key"]].head(3).to_dict("records"))

    print("\nOverlap checks:")
    print(
        "Watched ∩ diary:",
        len(set(watched["film_key"]) & set(diary["film_key"]))
    )
    print(
        "Watched ∩ reviews:",
        len(set(watched["film_key"]) & set(reviews["film_key"]))
    )

    watched = watched.drop_duplicates(subset="film_key", keep="last").copy()

    ratings = ratings.drop_duplicates(subset="film_key", keep="last").copy()
    ratings = ratings[["film_key", "Rating"]].rename(
        columns={"Rating": "letterboxd_rating"}
    )

    review_columns = ["film_key", "Review"]
    if "Rating" in reviews.columns:
        review_columns.append("Rating")

    reviews = reviews[review_columns].copy()
    reviews["Review"] = reviews["Review"].map(clean_text)
    reviews = reviews.dropna(subset=["Review"])
    reviews = reviews.groupby("film_key", as_index=False).agg(
        review_text=("Review", lambda values: "\n\n---\n\n".join(values)),
    )

    diary["Watched Date"] = pd.to_datetime(
        diary["Watched Date"],
        errors="coerce",
    )

    diary_dates = diary.groupby("film_key", as_index=False).agg(
        watch_dates=(
            "Watched Date",
            lambda values: sorted(
                {
                    value.strftime("%Y-%m-%d")
                    for value in values.dropna()
                }
            ),
        ),
    )

    merged = watched[
        ["film_key", "Name", "Year", "letterboxd_uri"]
    ].rename(
        columns={
            "Name": "title",
            "Year": "year",
        }
    )

    merged["watched"] = True

    merged = merged.merge(ratings, on="film_key", how="left")
    merged = merged.merge(diary_dates, on="film_key", how="left")
    merged = merged.merge(reviews, on="film_key", how="left")

    merged["year"] = pd.to_numeric(
        merged["year"],
        errors="coerce",
    ).astype("Int64")

    merged["letterboxd_rating"] = pd.to_numeric(
        merged["letterboxd_rating"],
        errors="coerce",
    )

    merged["watch_dates"] = merged["watch_dates"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    merged["review_text"] = merged["review_text"].map(clean_text)

    merged = merged[
        [
            "film_key",
            "letterboxd_uri",
            "title",
            "year",
            "watched",
            "letterboxd_rating",
            "watch_dates",
            "review_text",
        ]
    ].sort_values(
        by=["title", "year"],
        na_position="last",
    )

    csv_output = merged.copy()
    csv_output["watch_dates"] = csv_output["watch_dates"].apply(
        lambda dates: "|".join(dates)
    )
    csv_output.to_csv(OUTPUT_CSV, index=False)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as file:
        for record in merged.to_dict(orient="records"):
            record["year"] = (
                int(record["year"])
                if pd.notna(record["year"])
                else None
            )
            record["letterboxd_rating"] = (
                float(record["letterboxd_rating"])
                if pd.notna(record["letterboxd_rating"])
                else None
            )
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Created: {OUTPUT_CSV}")
    print(f"Created: {OUTPUT_JSONL}")
    print(f"Watched films: {len(merged)}")
    print(f"Films with ratings: {merged['letterboxd_rating'].notna().sum()}")
    print(f"Films with reviews: {merged['review_text'].notna().sum()}")
    print(f"Films with diary watch dates: {(merged['watch_dates'].str.len() > 0).sum()}")


if __name__ == "__main__":
    main()