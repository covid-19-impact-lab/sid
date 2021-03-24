from contextlib import ExitStack as does_not_raise  # noqa: N813

import pytest
from sid.validation import validate_function


@pytest.mark.parametrize(
    "x, expectation",
    [
        pytest.param(None, does_not_raise(), id="test with None"),
        pytest.param(lambda x: x, does_not_raise(), id="test with function"),
        pytest.param(
            1,
            pytest.raises(ValueError, match="must be a function or 'None'."),
            id="test with invalid input",
        ),
    ],
)
def test_validate_function(x, expectation):
    with expectation:
        validate_function(x, "lorem ipsum")
