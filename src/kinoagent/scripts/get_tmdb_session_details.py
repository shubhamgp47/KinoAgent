"""
One-time script to generate a permanent TMDb v3 session_id for your account,
WITHOUT ever typing your TMDb password into this script or your .env file.

How it works:
  1. This script asks TMDb for a temporary request_token (only needs your
     API key, which is a separate low-risk credential you already have
     in .env as TMDB_V3_API_KEY).
  2. It opens your web browser to TMDb's own login/approval page. You log
     in there directly on themoviedb.org and click "Approve" — your
     password is typed into TMDb's site, never into this script.
  3. Once you confirm you've approved it (by pressing Enter here), the
     script exchanges the now-approved token for a permanent session_id.
  4. It looks up your account_id using that session_id.

Run this ONCE:
    python get_tmdb_session.py

Then copy the two printed lines into your .env file:
    TMDB_SESSION_ID=...
    TMDB_ACCOUNT_ID=...

Requirements:
    pip install requests python-dotenv

Make sure .env already contains:
    TMDB_V3_API_KEY=your_v3_api_key
"""

import os
import sys
import webbrowser
import requests
from dotenv import load_dotenv

load_dotenv()

TMDB_V3_API_KEY = os.getenv("TMDB_V3_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

if not TMDB_V3_API_KEY:
    sys.exit(
        "TMDB_V3_API_KEY is not set. Add it to your .env first.\n"
        "Get one from https://www.themoviedb.org/settings/api"
    )


def get_request_token() -> str:
    resp = requests.get(
        f"{BASE_URL}/authentication/token/new",
        params={"api_key": TMDB_V3_API_KEY},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["request_token"]


def create_session(approved_token: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/authentication/session/new",
        params={"api_key": TMDB_V3_API_KEY},
        json={"request_token": approved_token},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["session_id"]


def get_account_id(session_id: str) -> int:
    resp = requests.get(
        f"{BASE_URL}/account",
        params={"api_key": TMDB_V3_API_KEY, "session_id": session_id},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def main():
    print("Step 1: requesting a temporary token from TMDb...")
    request_token = get_request_token()

    auth_url = f"https://www.themoviedb.org/authenticate/{request_token}"
    print(f"\nStep 2: opening your browser to approve the login:\n{auth_url}\n")
    webbrowser.open(auth_url)

    input(
        "Log in on the TMDb page that just opened, click 'Approve', "
        "then come back here and press Enter..."
    )

    print("\nStep 3: exchanging the approved token for a permanent session_id...")
    session_id = create_session(request_token)
    account_id = get_account_id(session_id)

    print("\nSuccess. Add these two lines to your .env file:\n")
    print(f"TMDB_SESSION_ID={session_id}")
    print(f"TMDB_ACCOUNT_ID={account_id}")


if __name__ == "__main__":
    main()