import csv

import pytest

from tools.run_ch3 import _csv_is_contiguous, main


def test_acceptance_runner_rejects_more_than_four_episodes_before_execution():
    with pytest.raises(SystemExit):
        main([
            "--phase",
            "acceptance",
            "--episodes",
            "5",
            "--max-steps",
            "20",
        ])


def test_resume_acceptance_requires_exact_one_to_four_csv(tmp_path):
    path = tmp_path / "episode_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode"])
        writer.writeheader()
        writer.writerows({"episode": value} for value in (1, 2, 3, 4))
    assert _csv_is_contiguous(path, 4)
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=["episode"]).writerow({"episode": 4})
    assert not _csv_is_contiguous(path, 4)
