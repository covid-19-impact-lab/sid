from contextlib import ExitStack as does_not_raise  # noqa: N813

import pandas as pd
import pytest
from sid.virus_strains import _factorize_boolean_infections
from sid.virus_strains import factorize_boolean_or_categorical_infections
from sid.virus_strains import factorize_categorical_infections


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
