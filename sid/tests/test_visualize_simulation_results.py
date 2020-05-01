from pathlib import Path

import pandas as pd
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


@pytest.mark.slow
def test_load_data_path(keep_vars):
    path = Path(__file__).resolve().parent / "simulation_results" / "001.pkl"
    expected_name = "001"
    expected_df = pd.read_pickle(path)[keep_vars]
    name, df = vsr._load_data(path, keep_vars=keep_vars, i=100)
    assert expected_name == name
    pd.testing.assert_frame_equal(expected_df, df)


@pytest.mark.slow
def test_load_data_df(keep_vars):
    path = Path(__file__).resolve().parent / "simulation_results" / "001.pkl"
    input_df = pd.read_pickle(path)
    expected_df = pd.read_pickle(path)[keep_vars]
    name, df = vsr._load_data(input_df, keep_vars=keep_vars, i=100)
    assert name == 100
    pd.testing.assert_frame_equal(expected_df, df)
