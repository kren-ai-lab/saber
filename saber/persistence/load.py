"""Persistence readers and artifact verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pandas.errors import EmptyDataError

from saber.datasets import FeatureSchema
from saber.exceptions import ArtifactIntegrityError
from saber.persistence.artifacts import LoadedBenchmarkArtifact, LoadedModelArtifact
from saber.persistence.checksums import verify_checksums
from saber.persistence.environment import compatibility_warnings
from saber.persistence.metadata import ArtifactManifest
from saber.utils.serialization import read_json


def inspect_artifact(
    path: str | Path,
    *,
    verify: bool = True,
) -> ArtifactManifest:
    """Read and validate a manifest without deserializing joblib content."""

    root = _artifact_root(path)
    if verify:
        verify_checksums(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactIntegrityError("Artifact manifest.json is missing.")
    return ArtifactManifest.from_dict(read_json(manifest_path))


def load_model_artifact(
    path: str | Path,
    *,
    verify: bool = True,
    strict_environment: bool = False,
) -> LoadedModelArtifact:
    """Verify and load a fitted model artifact.

    ``joblib`` uses Python pickle semantics; only load artifacts from trusted
    sources. Checksum verification protects integrity, not trust/authenticity.
    """

    root = _artifact_root(path)
    manifest = inspect_artifact(root, verify=verify)
    if manifest.artifact_type != "model":
        raise ArtifactIntegrityError(
            f"Expected model artifact, found '{manifest.artifact_type}'."
        )
    _require_manifest_files(root, manifest, ("model", "environment", "feature_schema", "provenance"))

    environment = read_json(root / manifest.files["environment"])
    warnings = compatibility_warnings(environment, strict=strict_environment)
    schema_payload = read_json(root / manifest.files["feature_schema"])
    schema = FeatureSchema(
        names=tuple(schema_payload["names"]),
        dtypes=tuple(schema_payload["dtypes"]),
    )
    if schema.fingerprint != schema_payload.get("fingerprint"):
        raise ArtifactIntegrityError("Feature-schema fingerprint is inconsistent.")

    provenance = read_json(root / manifest.files["provenance"])
    if provenance.get("feature_schema_fingerprint") != schema.fingerprint:
        raise ArtifactIntegrityError(
            "Model provenance and feature schema fingerprints do not match."
        )

    model = joblib.load(root / manifest.files["model"])
    return LoadedModelArtifact(
        path=root,
        model=model,
        manifest=manifest,
        feature_schema=schema,
        provenance=provenance,
        environment=environment,
        compatibility_warnings=warnings,
    )


def load_benchmark_artifact(
    path: str | Path,
    *,
    verify: bool = True,
    strict_environment: bool = False,
    load_object: bool = False,
) -> LoadedBenchmarkArtifact:
    """Load benchmark tables and optionally its trusted Python result object."""

    root = _artifact_root(path)
    manifest = inspect_artifact(root, verify=verify)
    if manifest.artifact_type != "benchmark":
        raise ArtifactIntegrityError(
            f"Expected benchmark artifact, found '{manifest.artifact_type}'."
        )
    _require_manifest_files(root, manifest, ("environment", "benchmark_metadata"))
    environment = read_json(root / manifest.files["environment"])
    warnings = compatibility_warnings(environment, strict=strict_environment)
    metadata = read_json(root / manifest.files["benchmark_metadata"])

    tables: dict[str, pd.DataFrame] = {}
    for name in ("runs", "metrics", "predictions", "failures", "optimization_history"):
        relative = manifest.files.get(name)
        if relative is not None:
            target = root / relative
            if not target.stat().st_size:
                tables[name] = pd.DataFrame()
            else:
                try:
                    tables[name] = pd.read_csv(target)
                except EmptyDataError:
                    tables[name] = pd.DataFrame()

    result = None
    object_file = manifest.files.get("benchmark_object")
    if load_object:
        if object_file is None:
            raise ArtifactIntegrityError(
                "Benchmark artifact does not contain a serialized Python object."
            )
        result = joblib.load(root / object_file)

    return LoadedBenchmarkArtifact(
        path=root,
        manifest=manifest,
        metadata=metadata,
        environment=environment,
        tables=tables,
        result=result,
        compatibility_warnings=warnings,
    )


def verify_artifact(path: str | Path) -> ArtifactManifest:
    """Verify checksums/schema and return the artifact manifest."""

    return inspect_artifact(path, verify=True)


def _artifact_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactIntegrityError(f"Artifact directory does not exist: '{root}'.")
    return root


def _require_manifest_files(
    root: Path,
    manifest: ArtifactManifest,
    names: tuple[str, ...],
) -> None:
    for name in names:
        relative = manifest.files.get(name)
        if relative is None:
            raise ArtifactIntegrityError(
                f"Artifact manifest does not define required file '{name}'."
            )
        if not (root / relative).is_file():
            raise ArtifactIntegrityError(
                f"Artifact required file is missing: '{relative}'."
            )
