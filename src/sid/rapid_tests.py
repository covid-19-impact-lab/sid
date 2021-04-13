import itertools
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
from sid.shared import boolean_choices
from sid.validation import validate_return_is_series_or_ndarray


def perform_rapid_tests(
    date: pd.Timestamp,
    states: pd.DataFrame,
    params: pd.DataFrame,
    rapid_test_models: Optional[Callable],
    contacts: pd.DataFrame,
    seed: itertools.count,
) -> pd.DataFrame:
    """Perform testing with rapid tests."""
    if rapid_test_models:
        receives_rapid_test = _compute_who_receives_rapid_tests(
            date=date,
            states=states,
            params=params,
            rapid_test_models=rapid_test_models,
            contacts=contacts,
            seed=seed,
        )

        is_tested_positive = _sample_test_outcome(
            states, receives_rapid_test, params, seed
        )

        states = _update_states_with_rapid_tests_outcomes(
            states, receives_rapid_test, is_tested_positive
        )

    return states


def apply_reactions_to_rapid_tests(
    date,
    states,
    params,
    rapid_test_reaction_models,
    contacts,
    seed,
):
    """Apply reactions to rapid_tests."""
    if rapid_test_reaction_models:
        for model in rapid_test_reaction_models.values():
            loc = model.get("loc", params.index)
            func = model["model"]

            if model["start"] <= date <= model["end"]:
                contacts = func(
                    contacts=contacts,
                    states=states,
                    params=params.loc[loc],
                    seed=next(seed),
                )

    return contacts


def _compute_who_receives_rapid_tests(
    date, states, params, rapid_test_models, contacts, seed
):
    """Compute who receives rapid tests.

    We loop over all rapid tests models and collect newly allocated rapid tests in
    ``receives_rapid_test``. A copy of the series is passed to each rapid test model so
    that the user is aware of who is tested, but cannot alter the existing assignment.

    """
    receives_rapid_test = pd.Series(index=states.index, data=False)

    for name, model in rapid_test_models.items():
        loc = model.get("loc", params.index)
        func = model["model"]

        if model["start"] <= date <= model["end"]:
            new_receives_rapid_test = func(
                receives_rapid_test=receives_rapid_test.copy(deep=True),
                states=states,
                params=params.loc[loc],
                contacts=contacts,
                seed=next(seed),
            )

            new_receives_rapid_test = validate_return_is_series_or_ndarray(
                new_receives_rapid_test, name, "rapid_test_models", states.index
            )

            receives_rapid_test.loc[new_receives_rapid_test] = True

    return receives_rapid_test


def _sample_test_outcome(states, receives_rapid_test, params, seed):
    """Sample the outcomes of the rapid tests.

    For those who are infectious, sensitivity gives us the probability that they are
    also tested positive.

    For those who are not infectious, 1 - specificity gives us the probability that they
    are falsely tested positive.

    """
    np.random.seed(next(seed))

    is_tested_positive = pd.Series(index=states.index, data=False)

    sensitivity = params.loc[("rapid_test", "sensitivity", "sensitivity"), "value"]
    receives_test_and_is_infectious = states["infectious"] & receives_rapid_test
    is_truly_positive = boolean_choices(
        np.full(receives_test_and_is_infectious.sum(), sensitivity)
    )

    specificity = params.loc[("rapid_test", "specificity", "specificity"), "value"]
    receives_test_and_is_not_infectious = ~states["infectious"] & receives_rapid_test
    is_falsely_positive = boolean_choices(
        np.full(receives_test_and_is_not_infectious.sum(), 1 - specificity)
    )

    is_tested_positive.loc[receives_test_and_is_infectious] = is_truly_positive
    is_tested_positive.loc[receives_test_and_is_not_infectious] = is_falsely_positive

    return is_tested_positive


def _update_states_with_rapid_tests_outcomes(
    states, receives_rapid_test, is_tested_positive
):
    """Updates states with outcomes of rapid tests."""
    states.loc[receives_rapid_test, "cd_received_rapid_test"] = 0
    states["is_tested_positive_by_rapid_test"] = is_tested_positive

    return states
