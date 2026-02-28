"""
download_ghcn.py — Download NOAA GHCN-Daily files for the weather pipeline.

Downloads 16 files (~1.4 GB total):
  - ghcnd-stations.txt         (~10 MB,  station metadata)
  - 2010.csv.gz … 2024.csv.gz  (~90 MB each, daily observations by year)

Usage
─────
    python download_ghcn.py                 # download all missing files
    python download_ghcn.py --years 2020 2021   # specific years only
    python download_ghcn.py --force         # re-download even if file exists
    python download_ghcn.py --check         # just print status, no download

Files are saved to:
    option1_weather_signal_detection/data/ghcn/
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

GHCN_DIR = Path(__file__).parent / "data" / "ghcn"

YEAR_MIN = 2010
YEAR_MAX = 2024

BASE_URL      = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
STATIONS_URL  = f"{BASE_URL}/ghcnd-stations.txt"
YEARLY_URL    = f"{BASE_URL}/by_year/{{year}}.csv.gz"

# Approximate sizes for progress display (bytes)
STATIONS_SIZE_APPROX  = 10 * 1024 * 1024       # ~10 MB
YEARLY_SIZE_APPROX    = 90 * 1024 * 1024        # ~90 MB each

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024:.0f} KB"


def _progress_hook(filename: str):
    """Return a reporthook for urllib.request.urlretrieve that prints progress."""
    start = time.time()

    def hook(block_count: int, block_size: int, total_size: int) -> None:
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            bar_len = 30
            filled = bar_len * pct // 100
            bar = "█" * filled + "░" * (bar_len - filled)
            elapsed = time.time() - start
            speed = downloaded / elapsed if elapsed > 0 else 0
            print(
                f"\r  [{bar}] {pct:3d}%  "
                f"{_fmt_bytes(downloaded)} / {_fmt_bytes(total_size)}  "
                f"({_fmt_bytes(int(speed))}/s)   ",
                end="",
                flush=True,
            )
        else:
            # total size unknown
            print(f"\r  {_fmt_bytes(downloaded)} downloaded…   ", end="", flush=True)

    return hook


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """
    Download url → dest.
    Returns True on success, False on failure.
    Skips if dest exists and force=False.
    """
    if dest.exists() and not force:
        size = dest.stat().st_size
        print(f"  ✓ already present ({_fmt_bytes(size)}) — skipping")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    print(f"  → {url}")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress_hook(dest.name))
        print()  # newline after progress bar
        tmp.rename(dest)
        print(f"  ✓ saved  {dest.name}  ({_fmt_bytes(dest.stat().st_size)})")
        return True
    except Exception as exc:
        print(f"\n  ✗ FAILED: {exc}")
        if tmp.exists():
            tmp.unlink()
        return False


def check_files(years: list[int]) -> tuple[list[str], list[str]]:
    """Return (present, missing) filename lists."""
    present, missing = [], []
    for name, path in _all_files(years):
        (present if path.exists() else missing).append(name)
    return present, missing


def _all_files(years: list[int]) -> list[tuple[str, Path]]:
    """Return [(display_name, Path)] for all files."""
    files = [("ghcnd-stations.txt", GHCN_DIR / "ghcnd-stations.txt")]
    for y in years:
        files.append((f"{y}.csv.gz", GHCN_DIR / f"{y}.csv.gz"))
    return files


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download NOAA GHCN-Daily files for the weather pipeline."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(YEAR_MIN, YEAR_MAX + 1)),
        metavar="YEAR",
        help=f"Years to download (default: {YEAR_MIN}–{YEAR_MAX})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check which files are present/missing, do not download",
    )
    args = parser.parse_args()

    years = sorted(args.years)
    total_files = 1 + len(years)   # stations + yearly files
    total_size_approx = STATIONS_SIZE_APPROX + YEARLY_SIZE_APPROX * len(years)

    print(f"\nNOAA GHCN-Daily download")
    print(f"  Destination : {GHCN_DIR.resolve()}")
    print(f"  Files       : {total_files}  ({_fmt_bytes(total_size_approx)} approx)")
    print(f"  Years       : {years[0]}–{years[-1]}")
    print()

    # ── Check mode ──
    if args.check:
        present, missing = check_files(years)
        print(f"Present ({len(present)}):")
        for f in present:
            print(f"  ✓ {f}")
        if missing:
            print(f"\nMissing ({len(missing)}):")
            for f in missing:
                print(f"  ✗ {f}")
            print()
        else:
            print("\nAll files present — ready to run 02_fetch_weather.py ✓\n")
        return 0 if not missing else 1

    # ── Download mode ──
    failures = []

    # 1. Station metadata
    print("─" * 60)
    print(f"[1/{total_files}]  ghcnd-stations.txt")
    ok = download_file(STATIONS_URL, GHCN_DIR / "ghcnd-stations.txt", force=args.force)
    if not ok:
        failures.append("ghcnd-stations.txt")

    # 2. Yearly files
    for i, year in enumerate(years, start=2):
        print(f"\n[{i}/{total_files}]  {year}.csv.gz")
        url  = YEARLY_URL.format(year=year)
        dest = GHCN_DIR / f"{year}.csv.gz"
        ok   = download_file(url, dest, force=args.force)
        if not ok:
            failures.append(f"{year}.csv.gz")

    # ── Summary ──
    print("\n" + "═" * 60)
    succeeded = total_files - len(failures)
    print(f"\nDone: {succeeded}/{total_files} files downloaded successfully.")

    if failures:
        print(f"\nFailed ({len(failures)}):")
        for f in failures:
            print(f"  ✗ {f}")
        print(
            "\nRe-run the script to retry failed files, "
            "or download them manually from:\n"
            f"  Stations : {STATIONS_URL}\n"
            f"  Yearly   : {BASE_URL}/by_year/YYYY.csv.gz\n"
        )
        return 1

    print(
        f"\nAll files saved to:\n  {GHCN_DIR.resolve()}\n\n"
        "Next step: run notebook 02\n"
        "  python option1_weather_signal_detection/02_fetch_weather.py\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
