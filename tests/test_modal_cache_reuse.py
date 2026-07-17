from modal_run import find_complete_cache_T


def _make_cache(root, T, prefix, indices):
    cache_dir = root / f"T{T}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        (cache_dir / f"{prefix}_{index:04d}.pt").touch()


def test_returns_none_without_a_complete_cache(tmp_path):
    _make_cache(tmp_path, 2048, "null", [0, 1])

    assert find_complete_cache_T(tmp_path, 1024, 3, "null") is None


def test_selects_smallest_complete_cache_at_least_requested_length(tmp_path):
    _make_cache(tmp_path, 1024, "null", range(3))
    _make_cache(tmp_path, 2048, "null", [0, 1])
    _make_cache(tmp_path, 8192, "null", range(3))

    assert find_complete_cache_T(tmp_path, 512, 3, "null") == 1024
    assert find_complete_cache_T(tmp_path, 1500, 3, "null") == 8192
    assert find_complete_cache_T(tmp_path, 8192, 3, "null") == 8192
    assert find_complete_cache_T(tmp_path, 8193, 3, "null") is None


def test_ignores_malformed_directories_and_other_prefixes(tmp_path):
    (tmp_path / "not-a-cache").mkdir()
    (tmp_path / "Tbad").mkdir()
    _make_cache(tmp_path, 4096, "wm", range(2))
    _make_cache(tmp_path, 8192, "null", range(2))

    assert find_complete_cache_T(tmp_path, 2048, 2, "null") == 8192
