"""Build paired validation and bounded-smoke manifests for efficiency v3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.ch3_efficiency_v3_registry import CH3_EFFICIENCY_V3_SCREEN
from tools.build_ch3_efficiency_scenarios import build_efficiency_manifest


V3_ROOT = PROJECT_ROOT / "data" / "chapter3_efficiency_v3_screen"
MANIFEST_ROOT = V3_ROOT / "manifests"
MANIFEST_SPECS = {
    "validation": ("efficiency_v3_screen_validation_scenarios.json", 50),
    "smoke": ("efficiency_v3_screen_smoke_scenarios.json", 2),
}


def build_v3_manifest(kind="validation"):
    if kind not in MANIFEST_SPECS:
        raise ValueError(f"unknown v3 manifest kind={kind!r}")
    _, count = MANIFEST_SPECS[kind]
    manifest = build_efficiency_manifest(
        "validation", count=count, generator_seed=41001,
        use_obstacles=False, obstacle_layout_id="none",
    )
    manifest.update(
        protocol=CH3_EFFICIENCY_V3_SCREEN,
        manifest_id=f"ch3_efficiency_v3_screen_{kind}_scenarios_v1",
        scenario_role=kind,
        max_steps=400,
    )
    for index, scenario in enumerate(manifest["scenarios"], 1):
        scenario["scenario_id"] = f"efficiency_v3_screen_{kind}_{index:04d}"
    return manifest


def candidate_config_diff_report():
    from registry.ch3_efficiency_v3_registry import (
        CH3_EFFICIENCY_V3_SCREEN_METHODS,
        config_diff,
        get_ch3_efficiency_v3_candidate,
        validate_v3_candidate_registry,
    )
    from train import CH3_EFFICIENCY_V2, get_ch3_method_config

    configs = validate_v3_candidate_registry()
    rows = {}
    for label in CH3_EFFICIENCY_V3_SCREEN_METHODS:
        entry = get_ch3_efficiency_v3_candidate(label)
        base = get_ch3_method_config(entry["base_method"], protocol=CH3_EFFICIENCY_V2)
        rows[label] = {
            **entry,
            "diff_from_v2_base": config_diff(base, configs[label]),
            "resolved_config": configs[label],
        }
    no_belief = configs["ch3_v3_no_belief_reference"]
    full = configs["ch3_v3_full_reference"]
    return {
        "protocol": CH3_EFFICIENCY_V3_SCREEN,
        "candidate_count": len(rows),
        "candidates": rows,
        "relative_diffs": {
            label: config_diff(
                full if label == "ch3_v3_gated_belief" else no_belief,
                configs[label],
            )
            for label in CH3_EFFICIENCY_V3_SCREEN_METHODS
            if label not in {"ch3_v3_full_reference", "ch3_v3_no_belief_reference"}
        },
    }


def write_v3_manifests(kinds=("validation", "smoke")):
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    for directory in ("runs", "summaries", "validation", "acceptance_smoke_runs"):
        (V3_ROOT / directory).mkdir(parents=True, exist_ok=True)
    outputs = {}
    for kind in kinds:
        filename, _ = MANIFEST_SPECS[kind]
        path = MANIFEST_ROOT / filename
        path.write_text(
            json.dumps(build_v3_manifest(kind), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        outputs[kind] = path
    diff_path = MANIFEST_ROOT / "v3_candidate_config_diffs.json"
    diff_path.write_text(
        json.dumps(candidate_config_diff_report(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outputs["config_diffs"] = diff_path
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("all", *MANIFEST_SPECS), default="all")
    args = parser.parse_args(argv)
    kinds = tuple(MANIFEST_SPECS) if args.kind == "all" else (args.kind,)
    for name, path in write_v3_manifests(kinds).items():
        print(f"[CH3 efficiency v3] wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
