import csv
import urllib.request
import re
import sys
import time


tmdbregex = r"data-tmdb-id=\"(\d+)\""
imdbregex = r"imdb.com/title/(tt\d+)"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def main(args):
    with open(args[0], mode="r", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)

        with open("tmdb-input.csv", mode="w", newline="", encoding="utf-8") as out_file:
            fieldnames = [
                "SIMKL_ID", "Title", "Type", "Year", "Watchlist",
                "LastEpWatched", "WatchedDate", "Rating", "Memo",
                "TVDB", "TMDB", "IMDB",
            ]
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()

            for row in csv_reader:
                print("Getting ID for movie:", row.get("Name", "?"), flush=True)
                try:
                    html = getHTML(row)
                except Exception as e:
                    print(f"  ERROR fetching {row.get('Name')}: {e}", flush=True)
                    html = ""

                tmdbid = getTMDBIDFromHTML(html)
                if tmdbid < 0:
                    print(f'  Could not find tmdb id for movie {row.get("Name")}', flush=True)
                imdbid = getIMDBIDFromHTML(html)
                print("  tmdb:", tmdbid, "imdb:", imdbid, flush=True)

                writer.writerow({
                    "SIMKL_ID": "", "Title": row.get("Name", ""), "Type": "",
                    "Year": row.get("Year", ""), "Watchlist": "",
                    "LastEpWatched": "", "WatchedDate": row.get("Date", ""),
                    "Rating": "", "Memo": "", "TVDB": "",
                    "TMDB": tmdbid, "IMDB": imdbid,
                })
                out_file.flush()
                time.sleep(0.5)  # be polite, avoid rate limiting / blocks


def getHTML(row):
    uri = row.get("Letterboxd URI", "")
    if uri.startswith("https://"):
        req = urllib.request.Request(uri, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read().decode("utf-8", errors="ignore")
    return ""


def getTMDBIDFromHTML(htmlstring: str):
    match = re.search(tmdbregex, htmlstring)
    return int(match.group(1)) if match else -1


def getIMDBIDFromHTML(htmlstring: str):
    match = re.search(imdbregex, htmlstring)
    return match.group(1) if match else ""


if __name__ == "__main__":
    if len(sys.argv[1:]):
        main(sys.argv[1:])
    else:
        print("filename missing. usage:\n\npython main.py letterboxd-filename.csv")