import pandas as pd
import pytest
from resources import CONTACT_MODELS
from sid import get_simulate_func


@pytest.mark.end_to_end
def test_simulate_rapid_tests(params, initial_states, tmp_path):
    rapid_test_models = {
        "rapid_tests": {
            "model": lambda receives_rapid_test, states, params, seed: pd.Series(
                index=states.index, data=True
            )
        }
    }

    params.loc[("infection_prob", "standard", "standard"), "value"] = 0.5

    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=CONTACT_MODELS,
        path=tmp_path,
        rapid_test_models=rapid_test_models,
        seed=144,
    )

    result = simulate(params)

    time_series = result["time_series"].compute()
    last_states = result["last_states"].compute()

    for df in [time_series, last_states]:
        assert isinstance(df, pd.DataFrame)

    assert last_states["is_tested_positive_by_rapid_test"].any()
    assert (last_states["cd_received_rapid_test"] == -1).all()
