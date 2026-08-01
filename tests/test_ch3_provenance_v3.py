from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.audit_ch3_provenance as provenance_audit
import tools.run_ch3_efficiency_v2 as efficiency_runner
from utils.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    algorithm_source_files,
    algorithm_source_fingerprint,
    assert_algorithm_source_unchanged,
    capture_provenance_snapshot,
    repository_source_fingerprint,
    source_fingerprint,
)


REQUIRED_FILES = (
    "train.py",
    "env.py",
    "base_env.py",
    "ch3_config.py",
    "ch3_constants.py",
    "target_motion.py",
    "runtime.py",
    "training.py",
    "metrics.py",
    "utils/agents.py",
    "utils/networks.py",
    "utils/noise.py",
    "utils/misc.py",
    "utils/ch3_buffer.py",
    "utils/__init__.py",
    "tools/__init__.py",
    "tools/build_ch3_efficiency_scenarios.py",
    "tools/build_ch3_efficiency_v3_scenarios.py",
    "tools/build_ch3_pilot_scenarios.py",
    "tools/build_ch3_scenarios.py",
    "map/map_module.py",
    "registry/ch3_efficiency_v3_registry.py",
)
REQUIRED_DIRECTORIES = ("algorithms", "comm", "registry")


def fake_repository(tmp_path):
    root = tmp_path / "repository"
    for relative in REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\nVALUE = 1\n", encoding="utf-8")
    for directory in REQUIRED_DIRECTORIES:
        path = root / directory / "__init__.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {directory}\n", encoding="utf-8")
    extras = {
        "tests/test_x.py": "VALUE = 1\n",
        "README.md": "version one\n",
        "tools/aggregate_ch3_efficiency_v2.py": "VALUE = 1\n",
        "tools/aggregate_ch3_efficiency_v3_screen.py": "VALUE = 1\n",
        "tools/run_ch3_efficiency_v3_screen.py": "VALUE = 1\n",
        "evaluate_pse.py": "VALUE = 1\n",
    }
    for relative, text in extras.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("relative", "algorithm_changes"),
    [
        ("env.py", True),
        ("train.py", True),
        ("utils/ch3_buffer.py", True),
        ("tests/test_x.py", False),
        ("README.md", False),
        ("tools/aggregate_ch3_efficiency_v2.py", False),
        ("tools/aggregate_ch3_efficiency_v3_screen.py", False),
        ("tools/run_ch3_efficiency_v3_screen.py", False),
        ("evaluate_pse.py", False),
    ],
)
def test_algorithm_and_repository_fingerprint_scopes(
    tmp_path, relative, algorithm_changes
):
    root = fake_repository(tmp_path)
    algorithm_before = algorithm_source_fingerprint(root)
    repository_before = repository_source_fingerprint(root)
    path = root / relative
    path.write_text(path.read_text(encoding="utf-8") + "VALUE_2 = 2\n", encoding="utf-8")
    assert (algorithm_source_fingerprint(root) != algorithm_before) is algorithm_changes
    assert repository_source_fingerprint(root) != repository_before


def test_algorithm_allowlist_is_sorted_and_requires_every_file(tmp_path):
    root = fake_repository(tmp_path)
    files = algorithm_source_files(root)
    assert files == sorted(files)
    assert "algorithms/__init__.py" in files
    assert "comm/__init__.py" in files
    assert "utils/__init__.py" not in files
    assert "tools/__init__.py" not in files
    assert "registry/ch3_efficiency_v3_registry.py" in files
    assert "map/map_module.py" in files
    assert "tools/build_ch3_efficiency_v3_scenarios.py" in files
    assert "tools/run_ch3_efficiency_v3_screen.py" not in files
    assert "tools/aggregate_ch3_efficiency_v3_screen.py" not in files
    (root / "utils/noise.py").unlink()
    with pytest.raises(FileNotFoundError, match="utils/noise.py"):
        algorithm_source_files(root)


def test_snapshot_is_json_serializable_and_freezes_algorithm_identity(tmp_path):
    root = fake_repository(tmp_path)
    snapshot = capture_provenance_snapshot(root)
    json.dumps(snapshot, allow_nan=False)
    assert snapshot["provenance_schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert snapshot["algorithm_source_files"] == sorted(
        snapshot["algorithm_source_files"]
    )

    test_file = root / "tests/test_x.py"
    test_file.write_text("VALUE = 2\n", encoding="utf-8")
    assert_algorithm_source_unchanged(root, snapshot)
    assert snapshot["algorithm_source_fingerprint"] == algorithm_source_fingerprint(root)
    assert snapshot["repository_source_fingerprint"] != repository_source_fingerprint(root)

    env_file = root / "env.py"
    env_file.write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source changed during the run"):
        assert_algorithm_source_unchanged(root, snapshot)

    snapshot = capture_provenance_snapshot(root)
    train_file = root / "train.py"
    train_file.write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source changed during the run"):
        assert_algorithm_source_unchanged(root, snapshot)


def test_repository_fingerprint_excludes_generated_trees_and_legacy_alias(tmp_path):
    root = fake_repository(tmp_path)
    before = repository_source_fingerprint(root)
    for relative in ("data/generated.py", "logs/run.txt", ".pytest_cache/state.txt"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ignored\n", encoding="utf-8")
    assert repository_source_fingerprint(root) == before
    assert source_fingerprint(root) == before


def test_provenance_audit_classifies_without_rewriting_runs(tmp_path):
    runs = tmp_path / "runs"
    current_algorithm = algorithm_source_fingerprint(provenance_audit.PROJECT_ROOT)
    current_repository = repository_source_fingerprint(provenance_audit.PROJECT_ROOT)

    verified_dir = runs / "ch3_pheromone_prior" / "seed_1"
    verified_dir.mkdir(parents=True)
    verified_summary = verified_dir / "training_summary.json"
    verified_summary.write_text(
        json.dumps({
            "method": "ch3_pheromone_prior",
            "run_type": "controller_only",
            "checkpoint_path": "N/A",
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "algorithm_source_fingerprint": current_algorithm,
            "repository_source_fingerprint": current_repository,
        }),
        encoding="utf-8",
    )
    original_bytes = verified_summary.read_bytes()

    legacy_dir = runs / "ch3_pse_no_residual" / "seed_1"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "training_summary.json").write_text(
        json.dumps({
            "method": "ch3_pse_no_residual",
            "run_type": "controller_only",
            "checkpoint_path": "N/A",
            "source_fingerprint": "legacy-repository-only",
        }),
        encoding="utf-8",
    )
    missing_dir = runs / "ch3_pse_rmaddpg" / "seed_2"
    missing_dir.mkdir(parents=True)
    mismatch_dir = runs / "ch3_pheromone_prior" / "seed_3"
    mismatch_dir.mkdir(parents=True)
    (mismatch_dir / "training_summary.json").write_text(
        json.dumps({
            "method": "ch3_pheromone_prior",
            "run_type": "controller_only",
            "checkpoint_path": "N/A",
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "algorithm_source_fingerprint": "different-algorithm",
            "repository_source_fingerprint": current_repository,
        }),
        encoding="utf-8",
    )

    report = provenance_audit.audit_runs(runs)
    assert report["existing_method_count"] == 3
    assert report["classification_counts"]["verified_v3"] == 1
    assert report["classification_counts"]["legacy_repository_only"] == 1
    assert report["classification_counts"]["missing_summary"] == 1
    assert report["classification_counts"]["algorithm_mismatch"] == 1
    assert report["formal_aggregation_eligible_count"] == 1
    assert verified_summary.read_bytes() == original_bytes

    output = tmp_path / "audit.json"
    assert provenance_audit.main([
        "--runs-root", str(runs), "--output", str(output)
    ]) == 0
    assert output.is_file()
    assert verified_summary.read_bytes() == original_bytes
    phase_output = tmp_path / "phase_audit.json"
    assert efficiency_runner.main([
        "--phase", "provenance-audit",
        "--output-dir", str(runs),
        "--audit-output", str(phase_output),
    ]) == 0
    assert phase_output.is_file()


def test_runner_accepts_default_and_algorithm_scoped_runs_roots(tmp_path, monkeypatch):
    v2_root = tmp_path / "chapter3_efficiency_v2"
    monkeypatch.setattr(efficiency_runner, "V2_ROOT", v2_root)
    default_runs = v2_root / "runs"
    scoped_runs = v2_root / "runs_by_algorithm" / "abcdef123456"
    default_runs.mkdir(parents=True)
    scoped_runs.mkdir(parents=True)

    assert efficiency_runner._require_runs_root(
        default_runs, label="aggregate runs root"
    ) == default_runs.resolve()
    assert efficiency_runner._require_runs_root(
        scoped_runs, label="aggregate runs root"
    ) == scoped_runs.resolve()
    with pytest.raises(ValueError, match="must be below one of"):
        efficiency_runner._require_runs_root(
            tmp_path / "outside", label="aggregate runs root"
        )
