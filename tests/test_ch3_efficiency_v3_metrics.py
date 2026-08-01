from train import summarize_evaluation_rows


def test_penalized_time_summary_is_unconditional():
    rows = [
        {"found": 1, "success": 1, "found_step": 120, "success_step": 150,
         "execution_delay": 30, "penalized_completion_step": 150,
         "normalized_penalized_completion": 0.375, "penalized_found_step": 120,
         "completion_failure": 0, "search_failure": 0},
        {"found": 0, "success": 0, "found_step": "", "success_step": "",
         "execution_delay": "", "penalized_completion_step": 500,
         "normalized_penalized_completion": 1.25, "penalized_found_step": 500,
         "completion_failure": 1, "search_failure": 1},
    ]
    result = summarize_evaluation_rows(rows)
    assert result["mean_penalized_completion_step"] == 325
    assert result["mean_normalized_penalized_completion"] == 0.8125
    assert result["mean_penalized_found_step"] == 310
    assert result["completion_failure_rate"] == 0.5
    assert result["search_failure_rate"] == 0.5
