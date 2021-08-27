from contextlib import ExitStack as does_not_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from sid.virus_strains import _factorize_boolean_infections
from sid.virus_strains import factorize_boolean_or_categorical_infections
from sid.virus_strains import factorize_categorical_infections
from sid.virus_strains import prepare_virus_strain_factors


@pytest.mark.unit
@pytest.mark.parametrize(
    "virus_strains, params, expectation, expected",
    [
        pytest.param(
            {"names": ["base_strain"]},
            None,
            does_not_raise(),
            {
                "names": ["base_strain"],
                "contagiousness_factor": np.ones(1),
                "immunity_resistance_factor": np.ones(1),
            },
            id="default single strain",
        ),
        pytest.param(
            {"names": ["b117"]},
            None,
            does_not_raise(),
            {
                "names": ["b117"],
                "contagiousness_factor": np.ones(1),
                "immunity_resistance_factor": np.ones(1),
            },
            id="non-default single strain",
        ),
        pytest.param(
            {"names": ["base_strain", "minus_strain"]},
            pd.DataFrame(
                {
                    "category": ["virus_strain"] * 4,
                    "subcategory": ["base_strain", "minus_strain"] * 2,
                    "name": ["contagiousness_factor"] * 2
                    + ["immunity_resistance_factor"] * 2,
                    "value": [1, -1, 1, 1],
                }
            ).set_index(["category", "subcategory", "name"]),
            pytest.raises(ValueError, match="Factors of 'virus_strains' cannot"),
            None,
            id="negative factor",
        ),
        pytest.param(
            {"names": ["base_strain"]},
            pd.DataFrame(
                {
                    "category": ["virus_strain"] * 2,
                    "subcategory": ["base_strain"] * 2,
                    "name": ["contagiousness_factor", "immunity_resistance_factor"],
                    "value": [0.5, 1.0],
                }
            ).set_index(["category", "subcategory", "name"]),
            does_not_raise(),
            {
                "names": ["base_strain"],
                "contagiousness_factor": np.ones(1),
                "immunity_resistance_factor": np.ones(1),
            },
            id="single factor stays the same if one",
        ),
        pytest.param(
            {"names": ["a_new_strain", "base_strain"]},
            pd.DataFrame(
                {
                    "category": ["virus_strain"] * 4,
                    "subcategory": ["base_strain", "a_new_strain"] * 2,
                    "name": ["contagiousness_factor"] * 2
                    + ["immunity_resistance_factor"] * 2,
                    "value": [0.5, 0.25, 1, 1],
                }
            ).set_index(["category", "subcategory", "name"]),
            does_not_raise(),
            {
                "names": ["a_new_strain", "base_strain"],
                "contagiousness_factor": np.array([0.5, 1]),
                "immunity_resistance_factor": np.array([1.0, 1]),
            },
            id="factors are scaled",
        ),
    ],
)
def test_prepare_virus_strain_factors(virus_strains, params, expectation, expected):
    with expectation:
        result = prepare_virus_strain_factors(virus_strains, params)
        assert all(np.array(result["names"]) == expected["names"])
        assert all(result["contagiousness_factor"] == expected["contagiousness_factor"])
        assert all(
            result["immunity_resistance_factor"]
            == expected["immunity_resistance_factor"]
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "virus_strain, names, expectation, expected_values, expected_categories",
    [
        (
            pd.Series(pd.Categorical([pd.NA, "0", "1"], categories=["0", "1"])),
            ["0", "1"],
            does_not_raise(),
            [-1, 0, 1],
            ["0", "1"],
        ),
        (
            pd.Series(pd.Categorical([pd.NA, "1", "0"], categories=["0", "1"])),
            ["0", "1"],
            does_not_raise(),
            [-1, 1, 0],
            ["0", "1"],
        ),
        (
            pd.Series(pd.Categorical([pd.NA, "2", "0"], categories=["0", "1", "2"])),
            ["0", "1", "2"],
            does_not_raise(),
            [-1, 2, 0],
            ["0", "1", "2"],
        ),
        (
            pd.Series(pd.Categorical([pd.NA, "0", "1"], categories=["0", "1"])),
            ["0"],
            pytest.raises(ValueError, match="Infections do not align"),
            None,
            None,
        ),
    ],
)
def test_factorize_categorical_infections(
    virus_strain, names, expectation, expected_values, expected_categories
):
    with expectation:
        values, categories = factorize_categorical_infections(virus_strain, names)

        assert (values == expected_values).all()
        assert (categories == expected_categories).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "infected, names, expectation, expected_values, expected_categories",
    [
        (
            pd.Series([True, False]),
            ["base_strain"],
            does_not_raise(),
            [0, -1],
            ["base_strain"],
        ),
        (
            pd.Series([True, True]),
            ["base_strain"],
            does_not_raise(),
            [0, 0],
            ["base_strain"],
        ),
        (
            pd.Series([False, False]),
            ["base_strain"],
            does_not_raise(),
            [-1, -1],
            ["base_strain"],
        ),
        (
            pd.Series([False]),
            ["0", "1"],
            pytest.raises(ValueError, match="Boolean infections"),
            None,
            None,
        ),
        (
            pd.Series([1]),
            ["0"],
            pytest.raises(ValueError, match="Infections must have a bool dtype."),
            None,
            None,
        ),
    ],
)
def test_factorize_boolean_infections(
    infected, names, expectation, expected_values, expected_categories
):
    with expectation:
        values, categories = _factorize_boolean_infections(infected, names)

        assert (values == expected_values).all()
        assert (categories == expected_categories).all()


@pytest.mark.integration
@pytest.mark.parametrize(
    "infections, virus_strains, expectation, expected_values",
    [
        (
            pd.Series([True, False]),
            {"names": ["base_strain"]},
            does_not_raise(),
            [0, -1],
        ),
        (
            pd.Series(
                pd.Categorical([pd.NA, "2", "0", "1"], categories=["0", "1", "2"])
            ),
            {"names": ["0", "1", "2"]},
            does_not_raise(),
            [-1, 2, 0, 1],
        ),
        (
            pd.Series([pd.NA, 0]),
            {"names": ["base_strain"]},
            pytest.raises(ValueError, match="Unknown dtype"),
            None,
        ),
        (
            pd.Series([1]),
            {"names": ["0"]},
            pytest.raises(ValueError, match="Unknown dtype of infections."),
            None,
        ),
        (
            pd.Series(pd.Categorical([pd.NA, "0", "1"], categories=["0", "1"])),
            {"names": ["0"]},
            pytest.raises(ValueError, match="Infections do not align"),
            None,
        ),
    ],
)
def test_factorize_boolean_or_categorical_infections(
    infections, virus_strains, expectation, expected_values
):
    with expectation:
        values = factorize_boolean_or_categorical_infections(infections, virus_strains)
        assert (values == expected_values).all()
