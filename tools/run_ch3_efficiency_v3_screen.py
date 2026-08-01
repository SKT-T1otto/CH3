"""Single entry point for Chapter-3 efficiency-v3 mechanism screening."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.ch3_efficiency_v3_registry import (  # noqa: E402
    CH3_EFFICIENCY_V3_SCREEN,
    CH3_EFFICIENCY_V3_SCREEN_METHODS,
    get_ch3_efficiency_v3_candidate,
    resolve_ch3_efficiency_v3_config,
)
from tools.aggregate_ch3_efficiency_v3_screen import (  # noqa: E402
    aggregate_v3_screen,
    validate_v3_run,
)
from tools.build_ch3_efficiency_v3_scenarios import (  # noqa: E402
    MANIFEST_ROOT,
    MANIFEST_SPECS,
    V3_ROOT,
    write_v3_manifests,
)
from train import (  # noqa: E402
    load_scenario_manifest,
    train_and_evaluate_resolved_config,
)
from utils.provenance import (  # noqa: E402
    algorithm_source_fingerprint,
    repository_source_fingerprint,
)


BUDGETS = {"smoke": 2, "pilot": 200}
CHECKPOINT_INTERVALS = {"smoke": 0, "pilot": 50}


def _device(value):
    if str(value).lower() != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def _below_v3(path, label):
    path = Path(path).resolve()
    try:
        path.relative_to(V3_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be below {V3_ROOT}, got {path}") from exc
    return path


def _manifest(kind):
    return MANIFEST_ROOT / MANIFEST_SPECS[kind][0]


def _validate_manifest(path, role):
    manifest, scenarios = load_scenario_manifest(path)
    checks = {
        "protocol": manifest.get("protocol") == CH3_EFFICIENCY_V3_SCREEN,
        "scenario_role": manifest.get("scenario_role") == role,
        "use_obstacles": manifest.get("use_obstacles") is False,
        "obstacle_layout_id": manifest.get("obstacle_layout_id") == "none",
        "flow_phases": all(
            float(row.get("flow_phase_x", 0.0)) == 0.0
            and float(row.get("flow_phase_y", 0.0)) == 0.0
            for row in scenarios
        ),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid efficiency-v3 {role} manifest: {failed}")
    return manifest, scenarios


def _manifest_role(path):
    manifest, _ = load_scenario_manifest(path)
    role = str(manifest.get("scenario_role", ""))
    if role not in {"validation", "smoke"}:
        raise ValueError(f"unsupported v3 scenario role={role!r}")
    return role


def _read_episode_numbers(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [int(row["episode"]) for row in csv.DictReader(handle)]


def _run_checked(name, command, timeout=900):
    print(f"[CH3 efficiency v3] {name}: {' '.join(map(str, command))}", flush=True)
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        timeout=float(timeout),
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with return code {result.returncode}")


def _completed_or_raise(
    method_dir,
    *,
    method,
    seed,
    episodes,
    max_steps,
    manifest_path,
    evaluation_limit=None,
    expected_replay_size=None,
):
    """Return True only for a complete, independently validated artifact set."""

    method_dir = Path(method_dir)
    summary_path = method_dir / "training_summary.json"
    occupied = method_dir.exists() and any(method_dir.iterdir())
    if not summary_path.is_file():
        if occupied:
            raise ValueError(
                f"refusing to overwrite incomplete/incompatible directory: {method_dir}"
            )
        return False
    _, scenarios = load_scenario_manifest(manifest_path)
    expected_scenarios = (
        len(scenarios)
        if evaluation_limit is None
        else min(len(scenarios), max(0, int(evaluation_limit)))
    )
    validate_v3_run(
        method,
        method_dir,
        seed=seed,
        expected_episodes=episodes,
        expected_max_steps=max_steps,
        expected_scenarios=expected_scenarios,
        scenario_manifest=manifest_path,
        expected_role=_manifest_role(manifest_path),
        expected_replay_size=expected_replay_size,
    )
    return True


def run_candidate(
    method,
    *,
    seed,
    episodes,
    max_steps,
    device,
    output_dir,
    manifest_path,
    resume=False,
    evaluation_limit=None,
    checkpoint_interval=0,
    replay_size=None,
):
    if method not in CH3_EFFICIENCY_V3_SCREEN_METHODS:
        raise ValueError(f"unknown candidate={method!r}")
    output_dir = _below_v3(output_dir, "output directory")
    role = _manifest_role(manifest_path)
    _validate_manifest(manifest_path, role)
    method_dir = output_dir / method / f"seed_{int(seed)}"

    if not resume:
        if _completed_or_raise(
            method_dir,
            method=method,
            seed=seed,
            episodes=episodes,
            max_steps=max_steps,
            manifest_path=manifest_path,
            evaluation_limit=evaluation_limit,
            expected_replay_size=replay_size,
        ):
            print(f"[CH3 efficiency v3] strict skip: {method_dir}", flush=True)
            return json.loads(
                (method_dir / "training_summary.json").read_text(encoding="utf-8")
            )
    else:
        resume_path = method_dir / "resume_state.pt"
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume state does not exist: {resume_path}")
        resume_state = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        completed_state_episode = int(resume_state.get("episode", -1))
        if int(episodes) <= completed_state_episode:
            raise ValueError(
                "resume target episodes must exceed the stored resume state: "
                f"target={episodes}, completed={completed_state_episode}"
            )
        summary_path = method_dir / "training_summary.json"
        if summary_path.is_file():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            completed = int(existing.get("episodes", -1))
            if int(episodes) <= completed:
                raise ValueError(
                    "resume target episodes must exceed the completed run: "
                    f"target={episodes}, completed={completed}"
                )
            _completed_or_raise(
                method_dir,
                method=method,
                seed=seed,
                episodes=completed,
                max_steps=max_steps,
                manifest_path=manifest_path,
                evaluation_limit=evaluation_limit,
                expected_replay_size=replay_size,
            )

    entry = get_ch3_efficiency_v3_candidate(method)
    config = resolve_ch3_efficiency_v3_config(method)
    metadata = {
        "candidate_label": method,
        "base_method": entry["base_method"],
        "config_overrides": entry["config_overrides"],
        "changed_mechanisms": entry["changed_mechanisms"],
        "screening_role": entry["screening_role"],
    }
    summary, _, _ = train_and_evaluate_resolved_config(
        method,
        config,
        seed=seed,
        episodes=episodes,
        max_steps=max_steps,
        device=_device(device),
        output_dir=output_dir,
        pilot=False,
        scenario_manifest=manifest_path,
        protocol=CH3_EFFICIENCY_V3_SCREEN,
        resume=resume,
        checkpoint_interval=checkpoint_interval,
        evaluation_limit=evaluation_limit,
        replay_size=replay_size,
        artifact_metadata=metadata,
    )
    print(
        f"[CH3 efficiency v3] completed method={method} "
        f"episodes={summary['episodes']} eval={summary['evaluation_scenarios']} "
        f"summary={method_dir / 'training_summary.json'}",
        flush=True,
    )
    return summary


def provenance_audit(runs_root, output_path, seed=1):
    """Read-only v3 audit that validates checkpoint, CSV and summary identity."""

    runs_root = Path(runs_root)
    current_algorithm = algorithm_source_fingerprint(PROJECT_ROOT)
    current_repository = repository_source_fingerprint(PROJECT_ROOT)
    rows = []
    for method in CH3_EFFICIENCY_V3_SCREEN_METHODS:
        method_dir = runs_root / method / f"seed_{int(seed)}"
        summary_path = method_dir / "training_summary.json"
        if not summary_path.is_file():
            rows.append({
                "method": method,
                "status": "missing",
                "formal_aggregation_eligible": False,
                "reasons": ["training_summary.json is missing"],
            })
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            scenario_manifest = Path(str(summary.get("scenario_manifest", "")))
            if not scenario_manifest.is_file():
                raise FileNotFoundError(
                    f"scenario manifest is missing: {scenario_manifest}"
                )
            role = _manifest_role(scenario_manifest)
            validate_v3_run(
                method,
                method_dir,
                seed=seed,
                expected_episodes=int(summary["episodes"]),
                expected_max_steps=int(summary["max_steps"]),
                expected_scenarios=int(summary["evaluation_scenarios"]),
                scenario_manifest=scenario_manifest,
                expected_role=role,
            )
            repository_matches = (
                summary.get("repository_source_fingerprint") == current_repository
            )
            rows.append({
                "method": method,
                "status": (
                    "verified"
                    if repository_matches
                    else "repository_only_mismatch"
                ),
                "formal_aggregation_eligible": True,
                "reasons": (
                    []
                    if repository_matches
                    else [
                        "repository source differs while algorithm source remains identical"
                    ]
                ),
            })
        except Exception as exc:  # audit must classify rather than mutate/abort
            text = f"{type(exc).__name__}: {exc}"
            status = (
                "algorithm_mismatch"
                if "algorithm" in text.lower()
                else "malformed_artifact"
            )
            rows.append({
                "method": method,
                "status": status,
                "formal_aggregation_eligible": False,
                "reasons": [text],
            })

    report = {
        "protocol": CH3_EFFICIENCY_V3_SCREEN,
        "algorithm_source_fingerprint": current_algorithm,
        "repository_source_fingerprint": current_repository,
        "complete_verified": all(
            row["formal_aggregation_eligible"] for row in rows
        ),
        "verified_count": sum(
            bool(row["formal_aggregation_eligible"]) for row in rows
        ),
        "candidates": rows,
    }
    output_path = _below_v3(output_path, "audit output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[CH3 efficiency v3] provenance audit "
        f"verified={report['verified_count']}/{len(rows)} output={output_path}",
        flush=True,
    )
    return report


def run_acceptance_smoke(args):
    _run_checked(
        "base acceptance gate",
        [
            sys.executable,
            "tools/run_ch3.py",
            "--phase",
            "acceptance",
            "--profiles",
            "all",
            "--base-candidate",
            "ch3_v3_full_reference",
            "--seed",
            "1",
            "--episodes",
            "3",
            "--resume-split",
            "1",
            "--max-steps",
            "20",
            "--replay-size",
            "32",
            "--checkpoint-interval",
            "1",
            "--evaluation-limit",
            "1",
            "--device",
            str(args.device),
            "--restart",
        ],
        timeout=900,
    )
    outputs = write_v3_manifests()
    print(
        "[CH3 efficiency v3] manifests: "
        + ", ".join(f"{key}={path}" for key, path in outputs.items()),
        flush=True,
    )

    gate_id = datetime.now(timezone.utc).strftime("gate_%Y%m%dT%H%M%SZ")
    smoke_root = _below_v3(
        V3_ROOT / "acceptance_smoke_runs" / gate_id,
        "smoke root",
    )
    smoke_max_steps = min(int(args.max_steps), 40)
    smoke_manifest = _manifest("smoke")
    for method in CH3_EFFICIENCY_V3_SCREEN_METHODS:
        run_candidate(
            method,
            seed=args.seed,
            episodes=2,
            max_steps=smoke_max_steps,
            device=args.device,
            output_dir=smoke_root,
            manifest_path=smoke_manifest,
            evaluation_limit=2,
            checkpoint_interval=0,
        )

    aggregate = aggregate_v3_screen(
        smoke_root,
        V3_ROOT / "summaries" / gate_id,
        seed=args.seed,
        expected_episodes=2,
        expected_max_steps=smoke_max_steps,
        expected_scenarios=2,
        scenario_manifest=smoke_manifest,
        expected_role="smoke",
    )

    first = CH3_EFFICIENCY_V3_SCREEN_METHODS[0]
    run_candidate(
        first,
        seed=args.seed,
        episodes=4,
        max_steps=smoke_max_steps,
        device=args.device,
        output_dir=smoke_root,
        manifest_path=smoke_manifest,
        resume=True,
        evaluation_limit=2,
        checkpoint_interval=0,
    )
    episode_numbers = _read_episode_numbers(
        smoke_root / first / f"seed_{int(args.seed)}" / "episode_metrics.csv"
    )
    if episode_numbers != [1, 2, 3, 4]:
        raise RuntimeError(
            f"v3 2->4 resume is not contiguous: {episode_numbers}"
        )

    audit = provenance_audit(
        smoke_root,
        V3_ROOT / "validation" / f"{gate_id}_provenance_audit_seed_{args.seed}.json",
        seed=args.seed,
    )
    if not audit["complete_verified"]:
        raise RuntimeError("v3 acceptance provenance audit did not verify all candidates")
    print(
        f"[CH3 efficiency v3] acceptance-smoke PASS gate={gate_id} "
        f"aggregate={aggregate['output_files']['aggregate_json']} "
        f"resume={episode_numbers}",
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "generate",
            "acceptance-smoke",
            "train",
            "aggregate",
            "provenance-audit",
        ),
    )
    parser.add_argument("--method", choices=CH3_EFFICIENCY_V3_SCREEN_METHODS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--budget", choices=tuple(BUDGETS), default="smoke")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--replay-size", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args(argv)

    if args.phase == "generate":
        for name, path in write_v3_manifests().items():
            print(f"[CH3 efficiency v3] wrote {name}: {path}")
        return 0

    episodes = int(
        args.episodes if args.episodes is not None else BUDGETS[args.budget]
    )
    checkpoint_interval = int(
        args.checkpoint_interval
        if args.checkpoint_interval is not None
        else CHECKPOINT_INTERVALS[args.budget]
    )
    if args.phase == "train":
        if not args.method:
            parser.error("--method is required for --phase train")
        run_candidate(
            args.method,
            seed=args.seed,
            episodes=episodes,
            max_steps=args.max_steps,
            device=args.device,
            output_dir=args.output_dir or V3_ROOT / "runs",
            manifest_path=args.manifest or _manifest("validation"),
            resume=args.resume,
            evaluation_limit=args.evaluation_limit,
            checkpoint_interval=checkpoint_interval,
            replay_size=args.replay_size,
        )
        return 0

    if args.phase == "aggregate":
        runs_root = _below_v3(
            args.output_dir or V3_ROOT / "runs", "runs root"
        )
        manifest_path = args.manifest or _manifest("validation")
        _, scenarios = _validate_manifest(manifest_path, _manifest_role(manifest_path))
        expected_scenarios = (
            len(scenarios)
            if args.evaluation_limit is None
            else min(len(scenarios), max(0, int(args.evaluation_limit)))
        )
        aggregate_v3_screen(
            runs_root,
            V3_ROOT / "summaries",
            seed=args.seed,
            expected_episodes=episodes,
            expected_max_steps=args.max_steps,
            expected_scenarios=expected_scenarios,
            scenario_manifest=manifest_path,
            expected_role=_manifest_role(manifest_path),
            expected_replay_size=args.replay_size,
            allow_partial=args.allow_partial,
        )
        return 0

    if args.phase == "provenance-audit":
        provenance_audit(
            _below_v3(args.output_dir or V3_ROOT / "runs", "runs root"),
            args.audit_output
            or V3_ROOT
            / "validation"
            / f"provenance_audit_seed_{args.seed}.json",
            seed=args.seed,
        )
        return 0

    run_acceptance_smoke(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
