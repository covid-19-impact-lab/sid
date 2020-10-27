import itertools

import numpy as np
import pytest
from sid.testing_demand import calculate_demand_for_tests


@pytest.mark.integration
@pytest.mark.parametrize("ask_for_tests", [True, False])
def test_calculate_demand_for_tests(initial_states, params, ask_for_tests):
    testing_demand_models = {
        "dummy_model": {"model": lambda *x: np.full(len(initial_states), ask_for_tests)}
    }
    seed = itertools.count(0)

    demands_test, demands_test_reason = calculate_demand_for_tests(
        initial_states, testing_demand_models, params, "2020-01-01", seed
    )

    assert (demands_test == ask_for_tests).all()
    assert (
        demands_test_reason.eq("dummy_model").all()
        if ask_for_tests
        else demands_test_reason.isna().all()
    )
