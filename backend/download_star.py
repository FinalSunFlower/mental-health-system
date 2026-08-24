"""Download and verify the public Project STAR table used by exp12."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


URL = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/STAR.csv"
EXPECTED_SHA256 = "0e8b179ea3d883730b25008ca1293c4c8f886aa8d5d299c03ab6ff05d0123ae6"
TARGET = Path(__file__).resolve().parents[1] / "data" / "raw" / "psychology" / "star.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix(".csv.download")
    urllib.request.urlretrieve(URL, temporary)
    actual = _sha256(temporary)
    if actual != EXPECTED_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Project STAR checksum mismatch: {actual}")
    temporary.replace(TARGET)
    print(TARGET)


if __name__ == "__main__":
    main()
