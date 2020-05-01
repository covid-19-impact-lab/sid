import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

import sid.pathogenesis as pg


def test_draw_countdowns_single_row(params):
    states = pd.DataFrame(index=range(10))
    single_row = params.loc["cd_infectious_true"]
    res = pg._draw_countdowns(states=states, param_slice=single_row)
    expected = pd.Series(
        3, index=range(10), dtype=np.int32  # this is the value in the single_row
    )
    assert_series_equal(left=expected, right=res)


def test_draw_countdowns_deterministic():
    states = pd.DataFrame(index=range(10))
    params = pd.DataFrame()
    params["value"] = [0, 1]
    params["subcategory"] = ["all", "all"]
    params["name"] = [2, 5]
    params.set_index(["subcategory", "name"], inplace=True)
    res = pg._draw_countdowns(states=states, param_slice=params)
    expected = pd.Series(5, index=range(10), dtype=np.int32)
    assert_series_equal(left=expected, right=res)


def test_draw_countdowns_no_age_variance():
    states = pd.DataFrame(index=range(4))
    params = pd.DataFrame()
    params["value"] = [0.4, 0.5, 0.1]
    params["name"] = [-1, 1, 2]
    params["subcategory"] = "all"
    params.set_index(["subcategory", "name"], inplace=True)

    np.random.seed(42091)
    res = pg._draw_countdowns(states=states, param_slice=params)
    expected = pd.Series([1, 1, -1, 1], index=range(4), dtype=np.int32)
    assert_series_equal(left=expected, right=res)


def test_draw_countdowns_age_variant():
    age_cats = ["young", "old", "young", "young", "old"]
    states = pd.Series(age_cats, name="age_group").to_frame()
    params = pd.DataFrame()
    params["value"] = [0.8, 0.2, 0.2, 0.3, 0.5]
    params["name"] = [-1, 1, 0, 2, 5]
    params["subcategory"] = ["young", "young", "old", "old", "old"]
    params.set_index(["subcategory", "name"], inplace=True)
    expected = pd.Series([-1, 5, -1, -1, 5], dtype=np.int32)

    np.random.seed(34981)
    res = pg._draw_countdowns(states=states, param_slice=params)
    assert_series_equal(left=expected, right=res)
