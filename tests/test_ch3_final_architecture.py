import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_TOOLS = (
    "tools/build_ch3_unknown_scenarios.py",
    "tools/validate_ch3_unknown.py",
    "tools/audit_ch3_unknown_provenance.py",
    "tools/audit_ch3_mission_provenance.py",
    "tools/run_ch3_unknown.py",
    "tools/run_ch3_schema4_acceptance.py",
    "tools/run_ch3_acceptance.py",
)


def _functions(path, name):
    tree = ast.parse((PROJECT_ROOT / path).read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]


def test_final_unified_architecture_has_single_public_flows():
    assert len(_functions("env.py", "step")) == 1
    assert len(_functions("base_env.py", "step")) == 0

    training_tree = ast.parse(
        (PROJECT_ROOT / "training.py").read_text(encoding="utf-8")
    )
    episode_loops = [
        node
        for node in ast.walk(training_tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
        and isinstance(node.target, ast.Name)
        and node.target.id == "episode"
    ]
    assert len(episode_loops) == 1

    assert len(_functions("runtime.py", "build_runtime")) == 1
    assert not _functions("runtime.py", "build_unknown_runtime")
    assert len(_functions("training.py", "train_and_evaluate")) == 1
    assert not _functions("training.py", "train_and_evaluate_unknown")
    assert not _functions("training.py", "run_unknown_episode")
    assert len(_functions("metrics.py", "augment_episode_metrics")) == 1
    assert not _functions("metrics.py", "augment_unknown_episode_metrics")

    assert len(_functions("tools/run_ch3.py", "_acceptance")) == 1
    assert not _functions("tools/run_ch3.py", "_acceptance_smoke")
    assert not _functions("tools/run_ch3.py", "_evaluate")
    assert not _functions("tools/run_ch3.py", "_command_stage")


def test_final_unified_tool_layout_and_old_symbols_are_absent():
    for relative in LEGACY_TOOLS:
        assert not (PROJECT_ROOT / relative).exists()

    active_files = (
        PROJECT_ROOT / "runtime.py",
        PROJECT_ROOT / "training.py",
        PROJECT_ROOT / "metrics.py",
        PROJECT_ROOT / "tools" / "run_ch3.py",
        PROJECT_ROOT / "tools" / "validate_ch3.py",
        PROJECT_ROOT / "tools" / "build_ch3_scenarios.py",
        PROJECT_ROOT / "tools" / "audit_ch3_provenance.py",
    )
    forbidden = (
        "build_unknown_runtime",
        "train_and_evaluate_unknown",
        "run_unknown_episode",
        "augment_unknown_episode_metrics",
        "run_ch3_acceptance.py",
    )
    for path in active_files:
        text = path.read_text(encoding="utf-8")
        assert not any(symbol in text for symbol in forbidden)
