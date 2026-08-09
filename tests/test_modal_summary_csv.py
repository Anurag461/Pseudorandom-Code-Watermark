from modal_run import CSV_COLUMNS


def test_summary_schema_keeps_experiment_fields_first():
    assert CSV_COLUMNS[:8] == [
        "eta",
        "T",
        "n",
        "r value",
        "r setting",
        "t",
        "Target FPR",
        "Entropy Model",
    ]
    assert "Generation Model" in CSV_COLUMNS
