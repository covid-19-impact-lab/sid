from contextlib import ExitStack as does_not_raise  # noqa: N813

import pytest
from sid.validation import _validate_model_function


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_name, model_group, model, args, expectation",
    [
        pytest.param(
            "a",
            "group_a",
            lambda x: True,
            ["x"],
            does_not_raise(),
            id="Pass and all arguments matched.",
        ),
        pytest.param(
            "b",
            "group_b",
            lambda x, y: True,
            ["x"],
            does_not_raise(),
            id="Pass only subset of params matched.",
        ),
        pytest.param(
            "c",
            "group_c",
            None,
            None,
            pytest.raises(TypeError, match="The model 'c' of 'group_c' is not a "),
            id="error when model is not a callable.",
        ),
        pytest.param(
            "d",
            "group_d",
            lambda x: True,
            ["x", "y"],
            pytest.raises(ValueError, match="The model 'd' of 'group_d' is missing"),
            id="error when not all arguments are accepted by the function.",
        ),
    ],
)
def test__validate_model_function(model_name, model_group, model, args, expectation):
    with expectation:
        _validate_model_function(model_name, model_group, model, args)
