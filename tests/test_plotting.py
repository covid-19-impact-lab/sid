import holoviews as hv
import pandas as pd
import pytest
import sid
from resources import CONTACT_MODELS
from sid.plotting import plot_infection_rates_by_contact_models


@pytest.mark.end_to_end
def test_plot_infection_rates_by_contact_models(params, initial_states, tmp_path):
    simulate = sid.get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=CONTACT_MODELS,
        saved_columns={"other": ["channel_infected_by_contact"]},
        path=tmp_path,
        seed=144,
    )

    result = simulate(params)

    time_series = result["time_series"].compute()
    last_states = result["last_states"].compute()

    for df in [time_series, last_states]:
        assert isinstance(df, pd.DataFrame)
        assert set(df["channel_infected_by_contact"].cat.categories) == {
            "not_infected_by_contact",
            "standard",
        }

    heatmap = plot_infection_rates_by_contact_models(time_series)
    assert isinstance(heatmap, hv.element.raster.HeatMap)
