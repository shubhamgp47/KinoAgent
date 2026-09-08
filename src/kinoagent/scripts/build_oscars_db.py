from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    BASE_DIR
    / "data"
    / "oscar_dataset"
    / "the_oscar_award.csv"
)

OUTPUT_DB = (
    BASE_DIR
    / "data"
    / "oscars_db"
    / "oscars.db"
)


def clean_text(value: Any) -> str | None:
    """Trim text values and convert empty values into None."""
    if value is None:
        return None

    cleaned = str(value).strip()

    if not cleaned:
        return None

    return cleaned


def to_int(value: Any) -> int | None:
    """Convert a CSV number to an integer when present."""
    cleaned = clean_text(value)

    if cleaned is None:
        return None

    return int(cleaned)


def to_bool_int(value: Any) -> int:
    """Store the CSV winner flag as 1 for True and 0 for False."""
    cleaned = clean_text(value)

    if cleaned is None:
        return 0

    return int(cleaned.casefold() in {"true", "1", "yes"})


def read_csv_rows(path: Path) -> list[tuple[Any, ...]]:
    """Read and normalize compact Oscar CSV rows for SQLite insertion."""
    rows: list[tuple[Any, ...]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        expected_columns = {
            "year_film",
            "year_ceremony",
            "ceremony",
            "category",
            "canon_category",
            "name",
            "film",
            "winner",
        }

        if reader.fieldnames is None:
            raise ValueError("The CSV does not contain a header row.")

        actual_columns = {
            column.strip()
            for column in reader.fieldnames
            if column
        }

        missing_columns = expected_columns - actual_columns

        if missing_columns:
            raise ValueError(
                "CSV is missing expected columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append(
                    (
                        to_int(row["year_film"]),
                        to_int(row["year_ceremony"]),
                        to_int(row["ceremony"]),
                        clean_text(row["category"]),
                        clean_text(row["canon_category"]),
                        clean_text(row["name"]),
                        clean_text(row["film"]),
                        to_bool_int(row["winner"]),
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Could not parse CSV line {line_number}: {error}"
                ) from error

    return rows


def create_schema(connection: sqlite3.Connection) -> None:
    """Create the nomination table and indexes for fast Oscar lookups. Every nomination gets a primary key, and the winner flag is stored as an integer (0 or 1)."""
    connection.executescript(
        """
        CREATE TABLE oscars_nominations (
            id INTEGER PRIMARY KEY,
            year_film INTEGER,
            year_ceremony INTEGER,
            ceremony INTEGER,
            category TEXT,
            canon_category TEXT,
            nominee_name TEXT,
            film TEXT,
            winner INTEGER NOT NULL CHECK (winner IN (0, 1))
        );

        CREATE INDEX idx_oscars_film
        ON oscars_nominations(film COLLATE NOCASE);

        CREATE INDEX idx_oscars_nominee
        ON oscars_nominations(nominee_name COLLATE NOCASE);

        CREATE INDEX idx_oscars_year_film
        ON oscars_nominations(year_film);

        CREATE INDEX idx_oscars_year_ceremony
        ON oscars_nominations(year_ceremony);

        CREATE INDEX idx_oscars_canon_category
        ON oscars_nominations(canon_category COLLATE NOCASE);

        CREATE INDEX idx_oscars_winner
        ON oscars_nominations(winner);
        """
    )


def insert_rows(
    connection: sqlite3.Connection,
    rows: list[tuple[Any, ...]],
) -> None:
    """Insert every normalized CSV row into the nominations table."""
    connection.executemany(
        """
        INSERT INTO oscars_nominations (
            year_film,
            year_ceremony,
            ceremony,
            category,
            canon_category,
            nominee_name,
            film,
            winner
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def print_verification_queries(
    connection: sqlite3.Connection,
) -> None:
    """Print small checks confirming that import and winner flags work."""
    cursor = connection.cursor()

    total_rows = cursor.execute(
        "SELECT COUNT(*) FROM oscars_nominations"
    ).fetchone()[0]

    total_wins = cursor.execute(
        """
        SELECT COUNT(*)
        FROM oscars_nominations
        WHERE winner = 1
        """
    ).fetchone()[0]

    first_winners = cursor.execute(
        """
        SELECT
            year_ceremony,
            canon_category,
            nominee_name,
            film
        FROM oscars_nominations
        WHERE winner = 1
        ORDER BY ceremony, canon_category
        LIMIT 5
        """
    ).fetchall()

    print(f"Total nomination rows: {total_rows}")
    print(f"Total winning rows: {total_wins}")
    print("\nFirst five winner records:")

    for row in first_winners:
        print(
            f"- Ceremony year {row[0]} | "
            f"{row[1]} | "
            f"{row[2]} | "
            f"{row[3]}"
        )


def main() -> None:
    """Rebuild the local SQLite Oscars database from the compact CSV."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Oscar CSV not found: {INPUT_CSV}"
        )

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()
        print(f"Removed old database: {OUTPUT_DB}")

    rows = read_csv_rows(INPUT_CSV)

    with sqlite3.connect(OUTPUT_DB) as connection:
        create_schema(connection)
        insert_rows(connection, rows)
        print_verification_queries(connection)

    print(f"\nCreated database: {OUTPUT_DB}")


if __name__ == "__main__":
    main()