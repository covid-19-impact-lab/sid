from pathlib import Path

import pandas as pd
import pytest
from sid.visualize_simulation_results import _create_folders
from sid.visualize_simulation_results import _load_data
from sid.visualize_simulation_results import _nice_str
from sid.visualize_simulation_results import visualize_simulation_results


@pytest.fixture
def keep_vars():
    return ["immune", "n_has_infected", "cd_infectious_false"]


def test_nice_str():
    s = "hello_world"
    res = _nice_str(s)
    expected = "Hello World"
    assert res == expected


def test_nice_str_no_change():
    s = "Bye World"
    res = _nice_str(s)
    assert res == s


def test_create_folders(tmp_path):
    bg_vars = ["age_group", "gender", "sector", "region"]
    expected = [tmp_path / "general"] + [tmp_path / x for x in bg_vars]
    _create_folders(tmp_path, bg_vars)
    for path in expected:
        assert path.exists()


@pytest.mark.optional
def test_load_data_path(keep_vars):
    path = Path(__file__).resolve().parent / "simulation_results" / "1.parquet"
    expected_name = "1"
    expected_df = pd.read_parquet(path)[keep_vars]
    name, df = _load_data(path, keep_vars=keep_vars, i=100)
    assert expected_name == name
    pd.testing.assert_frame_equal(expected_df, df)


@pytest.mark.optional
def test_load_data_df(keep_vars):
    path = Path(__file__).resolve().parent / "simulation_results" / "1.parquet"
    input_df = pd.read_parquet(path)
    expected_df = pd.read_parquet(path)[keep_vars]
    name, df = _load_data(input_df, keep_vars=keep_vars, i=100)
    assert name == 100
    pd.testing.assert_frame_equal(expected_df, df)


@pytest.mark.optional
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_visualize_simulation_results(tmp_path):
    path = Path(__file__).resolve().parent / "simulation_results"
    data = [path / "1.parquet", path / "2.parquet"]
    bg_vars = ["age_group"]
    infection_vars = ["immune", "infectious"]
    # just check that this does not raise any Errors:
    visualize_simulation_results(
        data=data,
        outdir_path=tmp_path,
        infection_vars=infection_vars,
        background_vars=bg_vars,
        window_length=2,
    )
