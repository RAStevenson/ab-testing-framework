"""Fetch the Hillstrom e-mail experiment dataset to the local cache.

The dataset is Kevin Hillstrom's MineThatData E-Mail Analytics And Data
Mining Challenge (2008): 64,000 customers randomly assigned to a men's
e-mail campaign, a women's e-mail campaign, or no e-mail, with visit,
conversion, and spend outcomes over the following two weeks
(https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html).

Its publisher never stated a license, so this repository does NOT
redistribute the file. This script downloads it from the original,
still-live URL into data/hillstrom/ (gitignored), verifies the pinned
SHA-256, and refuses a file that does not match (keeping the rejected
bytes beside the cache for review; note the 60s timeout is per socket
operation, not a total-download deadline), so the committed
analysis can never silently describe different data. Downloading at run
time from the source is the same pattern TensorFlow Datasets and
causeinfer use for this dataset.

Run with: python scripts/run_fetch_hillstrom.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hashlib
import traceback
import urllib.request

URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
# recorded from the first verified fetch; an upstream change surfaces as a
# mismatch here rather than as silently different analysis results
EXPECTED_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
DEST = Path(__file__).resolve().parents[1] / "data" / "hillstrom" / "hillstrom.csv"


def main() -> int:
    if DEST.is_file():
        digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
        if digest == EXPECTED_SHA256:
            print(f"already cached and verified: {DEST}")
            return 0
        print(f"cached file has sha256 {digest}, expected {EXPECTED_SHA256}; refetching")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    partial_download = DEST.with_suffix(".tmp")
    print(f"downloading {URL}")
    try:
        with (
            urllib.request.urlopen(URL, timeout=60) as response,
            open(partial_download, "wb") as out,
        ):
            while chunk := response.read(1 << 16):
                out.write(chunk)
    except OSError:
        traceback.print_exc()
        partial_download.unlink(missing_ok=True)
        print(
            f"error: the download failed, so nothing was written to {DEST.parent}. "
            "Rerun this script once the site is reachable.",
            file=sys.stderr,
        )
        return 2

    try:
        digest = hashlib.sha256(partial_download.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA256:
            rejected = DEST.with_suffix(".rejected")
            partial_download.replace(rejected)
            print(
                f"error: downloaded file has sha256 {digest}, but this repo pins "
                f"{EXPECTED_SHA256}. The bytes are kept at {rejected} for "
                "review (the payload may also be an error page rather than the "
                "CSV); re-pin only after reviewing, because the committed "
                "analysis describes the pinned version.",
                file=sys.stderr,
            )
            return 2
        partial_download.replace(DEST)
    except OSError:
        traceback.print_exc()
        print(
            f"error: the download finished but could not be moved into place, so "
            f"{DEST} was not updated (a partial copy may remain at {partial_download}). "
            "Rerun this script.",
            file=sys.stderr,
        )
        return 2
    print(f"fetched and verified: {DEST} ({DEST.stat().st_size:,} bytes)")
    print(
        "data: Kevin Hillstrom, MineThatData E-Mail Analytics And Data "
        "Mining Challenge (2008). Not redistributed here; no license was "
        "ever stated by the publisher."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
