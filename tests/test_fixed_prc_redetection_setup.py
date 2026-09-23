"""Protect source selection for the single-block historical redetection setup."""
from pathlib import Path

import pytest

import fixed_prc_redetection_setup as setup


def test_scope_excludes_online_multiblock_cross_model_and_completed_settings():
    rows = setup.selected_rows()
    assert len(rows) == 18
    settings = {(float(r["eta"]), int(r["n"])) for _, r in rows}
    assert (.05, 400) not in settings and (.05, 448) not in settings
    assert (.20, 8192) not in settings
    assert all(r["T"] == r["n"] and r["PRC Construction"] == "fixed_prc"
               and r["Generation Model"] == r["Entropy Model"] == setup.MODEL for _, r in rows)


def test_source_path_cannot_escape_volume(tmp_path):
    for name in ("/etc/passwd", "../escape", "dir/../escape", "dir//file"):
        with pytest.raises(ValueError, match="inside its volume"):
            setup.safe_source({"volume": "data", "path": name}, {"data": tmp_path})
    assert setup.safe_source({"volume": "data", "path": "wm/wm_0000.pt"}, {"data": tmp_path}) == tmp_path / "wm/wm_0000.pt"


def test_rate_requires_full_integer_cohort():
    assert setup.rate("450/500 (90.0%)") == {"detected": 450, "count": 500}
    for value in ("90%", "skipped", "45/50 (90.0%)", "501/500 (100.2%)"):
        with pytest.raises(ValueError):
            setup.rate(value)


@pytest.mark.skipif(not (setup.OUT / "catalog/provenance.tsv").exists(), reason="local archived provenance not downloaded")
def test_legacy_interleaving_and_original_workspace_shards_are_preserved():
    requests = setup.build_requests(setup.OUT / "catalog/provenance.tsv")
    assert len(requests) == 20
    for request in requests:
        n = request["n"]
        records = request["records"]
        assert request["batch_size"] <= 125
        if request["eta"] == .10 and n in (256, 400):
            assert records[0]["file"]["original_path"].endswith("gens/gen_0000.pt")
            assert records[499]["file"]["original_path"].endswith("gens/gen_0998.pt")
            assert records[500]["file"]["original_path"].endswith("gens/gen_0001.pt")
            assert records[-1]["file"]["original_path"].endswith("gens/gen_0999.pt")
        if request["eta"] == .20 and n in (2048, 4096):
            assert records[249]["file"]["original_workspace"] == "eta-0-20-shard-1"
            assert records[250]["file"]["original_workspace"] == "eta-0-20-shard-2"
            assert records[749]["file"]["original_workspace"] == "eta-0-20-shard-1"
            assert records[750]["file"]["original_workspace"] == "eta-0-20-shard-2"
        if request["group"] == "replicates":
            assert "seed" in request["tag"]
            assert all(r["file"]["path"].startswith("_nulls/T512/") for r in records if r["source"] == "null")
