import pytest

import sid.visualize_simulation_results as vsr


@pytest.fixture
def keep_vars():
    return ["immune", "infection_counter", "cd_infectious_false"]


def test_nice_str():
    s = "hello_world"
    res = vsr._nice_str(s)
    expected = "Hello World"
    assert res == expected


def test_nice_str_no_change():
    s = "Bye World"
    res = vsr._nice_str(s)
    assert res == s


def test_create_folders(tmp_path):
    bg_vars = ["age_group", "gender", "sector", "region"]
    expected = [tmp_path / "general"] + [tmp_path / x for x in bg_vars]
    vsr._create_folders(tmp_path, bg_vars)
    for path in expected:
        assert path.exists()
