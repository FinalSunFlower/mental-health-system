"""Download the public illustrative JOBS II dataset used by mediation."""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen


URL = "https://vincentarelbundock.github.io/Rdatasets/csv/mediation/jobs.csv"
EXPECTED_SHA256 = "88d5cce12930f3d111597cf15b7c0cd3e7fa7f316f718de7024e147b9756fb9f"


def main() -> None:
    output = (
        Path(__file__).resolve().parents[1]
        / "data" / "raw" / "psychology" / "jobs_ii.csv"
    )
    request = Request(URL, headers={"User-Agent": "CuspNet-research/1.0"})
    with urlopen(request, timeout=120) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(
            f"JOBS II hash mismatch: expected {EXPECTED_SHA256}, received {digest}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    print(f"Saved {len(content)} bytes to {output}")


if __name__ == "__main__":
    main()
