"""Source and runtime provenance helpers for the merged Chapter-3 project.

Schema v4 records the post-merge source layout.  The public compatibility
functions for base, mission, and unknown-map artifacts are retained, but all
implementations now live in this module.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


PROVENANCE_SCHEMA_VERSION = 4

_REPOSITORY_SOURCE_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yml", ".yaml"}
_EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
_EXCLUDED_TOP_LEVEL = {"data", "logs"}

# Legacy v1/v2/v3 execution still enters through train.py/base_env.py, while
# the physical implementation now resides in env.py.  Both are intentionally
# fingerprinted until the legacy entry points are removed in a later cleanup.
_BASE_REQUIRED_FILES = (
    "train.py",
    "base_env.py",
    "env.py",
    "map/map_module.py",
    "utils/agents.py",
    "utils/networks.py",
    "utils/noise.py",
    "utils/misc.py",
    "utils/ch3_buffer.py",
    "tools/build_ch3_efficiency_scenarios.py",
    "tools/build_ch3_efficiency_v3_scenarios.py",
    "tools/build_ch3_pilot_scenarios.py",
)
_BASE_DIRECTORIES = ("algorithms", "comm", "registry")

# S-profile mission behavior beyond the legacy base scope.
_MISSION_EXTENSION_FILES = (
    "ch3_config.py",
    "ch3_constants.py",
    "target_motion.py",
    "runtime.py",
    "training.py",
    "metrics.py",
    "map/path_planner.py",
    "tools/build_ch3_scenarios.py",
)

# Unknown-map behavior now lives entirely in shared mission-scoped files.
# A protocol-specific domain separator still makes the unknown-map fingerprint
# distinct even though no additional source file is appended here.
_UNKNOWN_EXTENSION_FILES = ()


def _hash_relative_files(root: Path, relative_paths: Sequence[str]) -> str:
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for relative_text in relative_paths:
        relative = Path(relative_text)
        path = root / relative
        relative_bytes = relative.as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(4, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _collect_required_files(
    root: Path, required_files: Sequence[str], directories: Sequence[str]
) -> list[str]:
    root = Path(root).resolve()
    missing: list[str] = []
    selected: set[str] = set()
    for relative_text in required_files:
        path = root / relative_text
        if not path.is_file():
            missing.append(relative_text)
        else:
            selected.add(Path(relative_text).as_posix())
    for directory_text in directories:
        directory = root / directory_text
        if not directory.is_dir():
            missing.append(f"{directory_text}/")
            continue
        python_files = sorted(path for path in directory.rglob("*.py") if path.is_file())
        if not python_files:
            missing.append(f"{directory_text}/**/*.py")
            continue
        for path in python_files:
            relative = path.relative_to(root)
            if any(part in _EXCLUDED_PARTS for part in relative.parts):
                continue
            selected.add(relative.as_posix())
    if missing:
        raise FileNotFoundError(
            "required Chapter-3 algorithm source is missing: "
            + ", ".join(sorted(missing))
        )
    return sorted(selected)


def base_algorithm_source_files(root: Path) -> list[str]:
    return _collect_required_files(root, _BASE_REQUIRED_FILES, _BASE_DIRECTORIES)


def base_algorithm_source_fingerprint(root: Path) -> str:
    root = Path(root).resolve()
    return _hash_relative_files(root, base_algorithm_source_files(root))


def mission_algorithm_source_files(root: Path) -> list[str]:
    root = Path(root).resolve()
    extension = _collect_required_files(root, _MISSION_EXTENSION_FILES, ())
    return sorted(set(base_algorithm_source_files(root)) | set(extension))


def mission_algorithm_source_fingerprint(root: Path) -> str:
    root = Path(root).resolve()
    base = base_algorithm_source_fingerprint(root)
    extension = _hash_relative_files(root, sorted(_MISSION_EXTENSION_FILES))
    digest = hashlib.sha256()
    digest.update(b"ch3-mission-v1-schema4\0")
    digest.update(base.encode("ascii"))
    digest.update(b"\0")
    digest.update(extension.encode("ascii"))
    return digest.hexdigest()


def unknown_map_algorithm_source_files(root: Path) -> list[str]:
    root = Path(root).resolve()
    extension = _collect_required_files(root, _UNKNOWN_EXTENSION_FILES, ())
    return sorted(set(mission_algorithm_source_files(root)) | set(extension))


def unknown_map_algorithm_source_fingerprint(root: Path) -> str:
    root = Path(root).resolve()
    mission = mission_algorithm_source_fingerprint(root)
    extension = _hash_relative_files(root, sorted(_UNKNOWN_EXTENSION_FILES))
    digest = hashlib.sha256()
    digest.update(b"ch3-unknown-map-v1-schema4\0")
    digest.update(mission.encode("ascii"))
    digest.update(b"\0")
    digest.update(extension.encode("ascii"))
    return digest.hexdigest()


def algorithm_source_files(root: Path) -> list[str]:
    """Compatibility alias for the legacy base identity scope."""
    return base_algorithm_source_files(root)


def algorithm_source_fingerprint(root: Path) -> str:
    return base_algorithm_source_fingerprint(root)


def _iter_repository_source_files(root: Path) -> Iterable[Path]:
    root = Path(root).resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.parts and relative.parts[0] in _EXCLUDED_TOP_LEVEL:
            continue
        if path.suffix.lower() not in _REPOSITORY_SOURCE_SUFFIXES:
            continue
        yield path


def repository_source_fingerprint(root: Path) -> str:
    root = Path(root).resolve()
    relative_paths = [
        path.relative_to(root).as_posix()
        for path in _iter_repository_source_files(root)
    ]
    return _hash_relative_files(root, relative_paths)


def source_fingerprint(root: Path) -> str:
    return repository_source_fingerprint(root)


def capture_base_provenance_snapshot(root: Path) -> dict:
    root = Path(root).resolve()
    files = base_algorithm_source_files(root)
    fingerprint = _hash_relative_files(root, files)
    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm_source_fingerprint": fingerprint,
        "base_algorithm_source_fingerprint": fingerprint,
        "repository_source_fingerprint": repository_source_fingerprint(root),
        "algorithm_source_files": files,
        "base_algorithm_source_files": files,
    }


def capture_mission_provenance_snapshot(root: Path) -> dict:
    root = Path(root).resolve()
    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_algorithm_source_fingerprint": base_algorithm_source_fingerprint(root),
        "mission_algorithm_source_fingerprint": mission_algorithm_source_fingerprint(root),
        "repository_source_fingerprint": repository_source_fingerprint(root),
        "base_algorithm_source_files": base_algorithm_source_files(root),
        "mission_algorithm_source_files": mission_algorithm_source_files(root),
    }


def capture_unknown_map_provenance_snapshot(root: Path) -> dict:
    root = Path(root).resolve()
    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_algorithm_source_fingerprint": base_algorithm_source_fingerprint(root),
        "mission_algorithm_source_fingerprint": mission_algorithm_source_fingerprint(root),
        "unknown_map_algorithm_source_fingerprint": unknown_map_algorithm_source_fingerprint(root),
        "repository_source_fingerprint": repository_source_fingerprint(root),
        "base_algorithm_source_files": base_algorithm_source_files(root),
        "mission_algorithm_source_files": mission_algorithm_source_files(root),
        "unknown_map_algorithm_source_files": unknown_map_algorithm_source_files(root),
    }


def capture_provenance_snapshot(root: Path) -> dict:
    return capture_base_provenance_snapshot(root)


def _assert_snapshot(root: Path, snapshot: dict, field: str, current: str, label: str) -> str:
    if int(snapshot.get("provenance_schema_version", -1)) != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            f"{label} provenance schema mismatch: expected "
            f"{PROVENANCE_SCHEMA_VERSION}, got "
            f"{snapshot.get('provenance_schema_version')}"
        )
    expected = snapshot.get(field)
    if not isinstance(expected, str) or not expected:
        raise ValueError(f"provenance snapshot has no {field}")
    if current != expected:
        raise RuntimeError(
            f"{label} source changed during the run; "
            f"expected={expected}, current={current}"
        )
    return current


def assert_base_algorithm_source_unchanged(root: Path, snapshot: dict) -> str:
    return _assert_snapshot(
        root,
        snapshot,
        "algorithm_source_fingerprint",
        base_algorithm_source_fingerprint(root),
        "Chapter-3 base algorithm",
    )


def assert_mission_algorithm_source_unchanged(root: Path, snapshot: dict) -> str:
    return _assert_snapshot(
        root,
        snapshot,
        "mission_algorithm_source_fingerprint",
        mission_algorithm_source_fingerprint(root),
        "Chapter-3 mission algorithm",
    )


def assert_unknown_map_source_unchanged(root: Path, snapshot: dict) -> str:
    return _assert_snapshot(
        root,
        snapshot,
        "unknown_map_algorithm_source_fingerprint",
        unknown_map_algorithm_source_fingerprint(root),
        "Chapter-3 unknown-map algorithm",
    )


def assert_algorithm_source_unchanged(root: Path, snapshot: dict) -> str:
    return assert_base_algorithm_source_unchanged(root, snapshot)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file_sha256(path: Path) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def runtime_versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
