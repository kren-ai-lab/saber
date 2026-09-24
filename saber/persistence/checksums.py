"""SHA-256 checksums for persistence artifacts."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from saber.exceptions import ArtifactIntegrityError

CHECKSUM_FILENAME = "checksums.sha256"


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: str | Path) -> Path:
    base = Path(root)
    entries: list[tuple[str, str]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name == CHECKSUM_FILENAME:
            continue
        relative = path.relative_to(base).as_posix()
        entries.append((sha256_file(path), relative))

    output = base / CHECKSUM_FILENAME
    output.write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in entries),
        encoding="utf-8",
    )
    return output


def verify_checksums(root: str | Path) -> None:
    base = Path(root)
    checksum_path = base / CHECKSUM_FILENAME
    if not checksum_path.is_file():
        raise ArtifactIntegrityError(
            f"Artifact checksum file '{CHECKSUM_FILENAME}' is missing."
        )

    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            expected, relative = raw_line.split("  ", 1)
        except ValueError as exc:
            raise ArtifactIntegrityError(
                f"Malformed checksum entry on line {line_number}."
            ) from exc
        target = base / relative
        if not target.is_file():
            raise ArtifactIntegrityError(
                f"Artifact file listed in checksums is missing: '{relative}'."
            )
        observed = sha256_file(target)
        if observed != expected:
            raise ArtifactIntegrityError(
                f"Checksum mismatch for '{relative}': expected {expected}, observed {observed}."
            )
