"""
Download VGC replays from replay.pokemonshowdown.com.

Usage:
    from pokemon_rl.imitation.downloader import download_replays
    paths = download_replays(format_id="gen9vgc2025regg", n_replays=200)
"""

import json
import time
from pathlib import Path
from typing import List, Optional

import requests

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) VGC-RL-Research/1.0",
    "Accept": "application/json",
    "Referer": "https://replay.pokemonshowdown.com/",
}
_SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
_REPLAY_URL = "https://replay.pokemonshowdown.com/{replay_id}.json"


def _search_page(format_id: str, page: int) -> List[dict]:
    r = requests.get(
        _SEARCH_URL,
        params={"format": format_id, "page": page},
        headers=_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def _fetch_replay(replay_id: str) -> dict:
    r = requests.get(
        _REPLAY_URL.format(replay_id=replay_id),
        headers=_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def download_replays(
    format_id: str = "gen9vgc2025regg",
    n_replays: int = 100,
    output_dir: str = "data/replays",
    delay: float = 0.4,
    min_rating: Optional[int] = None,
) -> List[Path]:
    """
    Download up to n_replays replays for the given format.

    Parameters
    ----------
    format_id   : PS format string (e.g. "gen9vgc2025regg")
    n_replays   : Maximum number of replays to download
    output_dir  : Directory to save replay JSON files
    delay       : Seconds to wait between requests
    min_rating  : If set, skip replays below this rating

    Returns
    -------
    List of Paths to saved JSON files
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    page = 1

    while len(saved) < n_replays:
        try:
            entries = _search_page(format_id, page)
        except requests.RequestException as e:
            print(f"Search page {page} failed: {e}")
            break
        if not entries:
            break

        for entry in entries:
            if len(saved) >= n_replays:
                break

            # Filter by rating if requested
            rating = entry.get("rating")
            if min_rating is not None and (rating is None or rating < min_rating):
                continue

            replay_id = entry["id"]
            dest = out / f"{replay_id}.json"

            if dest.exists():
                saved.append(dest)
                continue

            try:
                data = _fetch_replay(replay_id)
                dest.write_text(json.dumps(data))
                saved.append(dest)
                print(f"  [{len(saved)}/{n_replays}] {replay_id}"
                      + (f"  rating={rating}" if rating else ""))
                time.sleep(delay)
            except requests.RequestException as e:
                print(f"  Failed {replay_id}: {e}")

        page += 1

    print(f"\nDownloaded {len(saved)} replays to {out}/")
    return saved


def load_replay_json(path: Path) -> dict:
    """Load a saved replay JSON file."""
    return json.loads(path.read_text())
