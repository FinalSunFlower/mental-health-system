"""Download and verify the public Tuebingen cause-effect pair benchmark."""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from app.data.paths import RAW_DIR


URL = "https://webdav.tuebingen.mpg.de/cause-effect/pairs.zip"
SHA256 = "C1BC9CD212B2ED1BC18C87D59761052478983623E3F2C4B4166260FA0A23FB11"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    target = Path(RAW_DIR) / "tuebingen" / "pairs.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and _sha256(target).upper() == SHA256:
        print(f"verified existing archive: {target}")
    else:
        temporary = target.with_suffix(".download")
        try:
            urllib.request.urlretrieve(URL, temporary)
            if _sha256(temporary).upper() != SHA256:
                raise RuntimeError("downloaded archive SHA-256 does not match the pinned value")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"downloaded and verified: {target}")
    extracted = target.with_suffix("")
    if extracted.exists():
        shutil.rmtree(extracted)
    with zipfile.ZipFile(target) as archive:
        archive.extractall(extracted)
    print(f"extracted: {extracted}")


if __name__ == "__main__":
    main()
