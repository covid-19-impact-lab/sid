from itertools import count

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from sid.config import DTYPE_DRAW_COURSE_OF_DISEASE
from sid.pathogenesis import _draw_countdowns
from sid.pathogenesis import draw_course_of_disease
from sid.shared import get_epidemiological_parameters


def test_draw_countdowns_single_row(params):
    states = pd.DataFrame(index=range(10))
    single_row = params.loc["cd_infectious_true"]
    res = _draw_countdowns(states, single_row)
    expected = pd.Series(
        3,
        index=range(10),
        dtype=DTYPE_DRAW_COURSE_OF_DISEASE,
    )
    assert_series_equal(left=expected, right=res)


def test_draw_countdowns_deterministic():
    states = pd.DataFrame(index=range(10))
    params = pd.DataFrame(
        {"value": [0, 1], "subcategory": ["all", "all"], "name": [2, 5]}
    ).set_index(["subcategory", "name"])
    res = _draw_countdowns(states, params)
    expected = pd.Series(5, index=range(10), dtype=DTYPE_DRAW_COURSE_OF_DISEASE)
    assert_series_equal(left=expected, right=res)


def test_draw_countdowns_no_age_variance():
    states = pd.DataFrame(index=range(4))
    params = pd.DataFrame(
        {"value": [0.4, 0.5, 0.1], "name": [-1, 1, 2], "subcategory": "all"}
    ).set_index(["subcategory", "name"])

    np.random.seed(42091)
    res = _draw_countdowns(states, params)
    expected = pd.Series(
        [1, 1, -1, 1], index=range(4), dtype=DTYPE_DRAW_COURSE_OF_DISEASE
    )
    assert_series_equal(left=expected, right=res)


def test_draw_countdowns_age_variant():
    states = pd.DataFrame({"age_group": ["young", "old", "young", "young", "old"]})
    params = pd.DataFrame(
        {
            "value": [0.8, 0.2, 0.2, 0.3, 0.5],
            "name": [-1, 1, 0, 2, 5],
            "subcategory": ["young", "young", "old", "old", "old"],
        }
    ).set_index(["subcategory", "name"])
    expected = pd.Series([-1, 5, -1, -1, 5], dtype=DTYPE_DRAW_COURSE_OF_DISEASE)

    np.random.seed(34981)
    res = _draw_countdowns(states, params)
    assert_series_equal(left=expected, right=res)


def test_draw_countdowns_with_covid_params():
    # This test makes sure that the provided covid params are not corrupted
    # It also tests that everyone scheduled to die won't recover before.
    age_groups = [
        "0-9",
        "10-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70-79",
        "80-100",
    ]
    states = pd.DataFrame({"age_group": age_groups * 10})
    params = get_epidemiological_parameters()
    counter = count()
    res = draw_course_of_disease(states, params, counter)
    to_die = res[res["cd_dead_true_draws"] > 0]
    assert (to_die["cd_needs_icu_false_draws"] == -1).all()
