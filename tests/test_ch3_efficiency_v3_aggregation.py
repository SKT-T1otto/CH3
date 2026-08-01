from tools.aggregate_ch3_efficiency_v3_screen import apply_shortlist, pareto_relations, rank_candidates


def _row(method, success, completion, found, found_step, energy, separation=0):
    return {"method": method, "success_rate": success,
            "mean_penalized_completion_step": completion, "found_rate": found,
            "mean_penalized_found_step": found_step, "energy_cost": energy,
            "minimum_separation_violation_rate": separation}


def test_lexicographic_rank_prioritizes_success_then_penalized_time():
    rows = [_row("a", .90, 100, 1, 80, 10), _row("b", .92, 300, .9, 200, 20), _row("c", .90, 90, 1, 70, 10)]
    ranked = rank_candidates(rows)
    assert [row["method"] for row in ranked] == ["b", "c", "a"]


def test_pareto_and_shortlist_rules_are_explicit():
    rows = [
        _row("ch3_v3_full_reference", .90, 200, .9, 150, 100),
        _row("ch3_v3_no_belief_reference", .91, 180, .92, 140, 90),
        _row("candidate", .91, 175, .92, 135, 92),
    ]
    pareto_relations(rows)
    apply_shortlist(rows, complete=True, provenance_verified=True)
    by_name = {row["method"]: row for row in rows}
    assert by_name["ch3_v3_no_belief_reference"]["pareto_front_member"]
    assert by_name["candidate"]["shortlist_status"] == "shortlisted"
    apply_shortlist(rows, complete=False, provenance_verified=True)
    assert {row["shortlist_status"] for row in rows} == {"insufficient_single_seed_evidence"}


def test_complete_v3_aggregate_validates_full_artifact_identity(tmp_path):
    import json
    from pathlib import Path

    import pytest

    from registry.ch3_efficiency_v3_registry import (
        CH3_EFFICIENCY_V3_SCREEN,
        CH3_EFFICIENCY_V3_SCREEN_METHODS,
        get_ch3_efficiency_v3_candidate,
        resolve_ch3_efficiency_v3_config,
    )
    from tools.aggregate_ch3_efficiency_v3_screen import aggregate_v3_screen
    from tools.build_ch3_efficiency_v3_scenarios import build_v3_manifest
    from train import train_and_evaluate_resolved_config

    manifest_path = tmp_path / "smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(build_v3_manifest("smoke"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    summaries = tmp_path / "summaries"

    for method in CH3_EFFICIENCY_V3_SCREEN_METHODS:
        entry = get_ch3_efficiency_v3_candidate(method)
        metadata = {
            "candidate_label": method,
            "base_method": entry["base_method"],
            "config_overrides": entry["config_overrides"],
            "changed_mechanisms": entry["changed_mechanisms"],
            "screening_role": entry["screening_role"],
        }
        train_and_evaluate_resolved_config(
            method,
            resolve_ch3_efficiency_v3_config(method),
            seed=1,
            episodes=1,
            max_steps=2,
            device="cpu",
            output_dir=runs,
            pilot=False,
            scenario_manifest=manifest_path,
            protocol=CH3_EFFICIENCY_V3_SCREEN,
            checkpoint_interval=0,
            evaluation_limit=1,
            replay_size=32,
            artifact_metadata=metadata,
        )

    report = aggregate_v3_screen(
        runs,
        summaries,
        seed=1,
        expected_episodes=1,
        expected_max_steps=2,
        expected_scenarios=1,
        scenario_manifest=manifest_path,
        expected_role="smoke",
        expected_replay_size=32,
        bootstrap_samples=8,
    )
    assert report["complete_method_set"] is True
    assert report["provenance_verified"] is True
    assert len(report["methods"]) == 6
    assert all(Path(path).is_file() for path in report["output_files"].values())

    method = CH3_EFFICIENCY_V3_SCREEN_METHODS[0]
    summary_path = runs / method / "seed_1" / "training_summary.json"
    original = json.loads(summary_path.read_text(encoding="utf-8"))

    tampered = dict(original)
    tampered["episodes"] = 2
    summary_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="v3 protocol mismatch"):
        aggregate_v3_screen(
            runs,
            summaries,
            seed=1,
            expected_episodes=1,
            expected_max_steps=2,
            expected_scenarios=1,
            scenario_manifest=manifest_path,
            expected_role="smoke",
            expected_replay_size=32,
            bootstrap_samples=8,
        )

    summary_path.write_text(json.dumps(original), encoding="utf-8")
    tampered = dict(original)
    tampered["checkpoint_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint SHA256"):
        aggregate_v3_screen(
            runs,
            summaries,
            seed=1,
            expected_episodes=1,
            expected_max_steps=2,
            expected_scenarios=1,
            scenario_manifest=manifest_path,
            expected_role="smoke",
            expected_replay_size=32,
            bootstrap_samples=8,
        )
