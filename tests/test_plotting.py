import holoviews as hv
import pandas as pd
import pytest
import sid
from resources import CONTACT_MODELS
from sid.plotting import plot_infection_rates_by_contact_models
from sid.plotting import plot_policy_gantt_chart


POLICIES_FOR_GANTT_CHART = {
    "closed_schools": {
        "affected_contact_model": "school",
        "start": "2020-03-09",
        "end": "2020-05-31",
        "policy": 0,
    },
    "partially_closed_schools": {
        "affected_contact_model": "school",
        "start": "2020-06-01",
        "end": "2020-09-30",
        "policy": 0.5,
    },
    "partially_closed_kindergarden": {
        "affected_contact_model": "school",
        "start": "2020-05-20",
        "end": "2020-06-30",
        "policy": 0.5,
    },
    "work_closed": {
        "affected_contact_model": "work",
        "start": "2020-03-09",
        "end": "2020-06-15",
        "policy": 0.4,
    },
    "work_partially_opened": {
        "affected_contact_model": "work",
        "start": "2020-05-01",
        "end": "2020-08-15",
        "policy": 0.7,
    },
    "closed_leisure_activities": {
        "affected_contact_model": "leisure",
        "start": "2020-03-09",
        "policy": 0,
    },
}


@pytest.mark.unit
def test_plot_policy_gantt_chart():
    plot_policy_gantt_chart(POLICIES_FOR_GANTT_CHART, effects=True)


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
