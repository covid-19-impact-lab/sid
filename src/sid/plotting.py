import dask.dataframe as dd
import holoviews as hv
import pandas as pd


ERROR_MISSING_CHANNEL = (
    "'channel_infected_by_contact' is necessary to plot infection rates by contact "
    "models. Re-run the simulation and pass `saved_columns={'channels': "
    "'channel_infected_by_contact'}` to `sid.get_simulate_func`."
)


DEFAULT_IR_PER_CM_KWARGS = {
    "width": 600,
    "height": 400,
    "tools": ["hover"],
    "title": "Contribution of Contact Models to Infections",
    "xlabel": "Date",
    "ylabel": "Contact Model",
    "invert_yaxis": True,
    "colorbar": True,
}


def plot_infection_rates_by_contact_models(time_series, fig_kwargs=None):
    """Plot infection rates by contact models."""
    if "channel_infected_by_contact" not in time_series:
        raise ValueError(ERROR_MISSING_CHANNEL)

    fig_kwargs = (
        DEFAULT_IR_PER_CM_KWARGS
        if fig_kwargs is None
        else {**DEFAULT_IR_PER_CM_KWARGS, **fig_kwargs}
    )

    if isinstance(time_series, pd.DataFrame):
        df = time_series[["date", "channel_infected_by_contact"]]
    elif isinstance(time_series, dd.core.DataFrame):
        df = time_series[["date", "channel_infected_by_contact"]].compute()
    else:
        raise ValueError("'time_series' must be either pd.DataFrame or dask.dataframe.")

    df = (
        df.groupby(["date", "channel_infected_by_contact"])
        .size()
        .reset_index()
        .rename(columns={0: "n"})
    )
    df["share"] = df["n"] / df.groupby(["date"])["n"].transform("sum")
    df = df.drop(columns="n")
    df = df.query("channel_infected_by_contact != 'not_infected_by_contact'")

    hv.extension("bokeh")

    heatmap = hv.HeatMap(df)
    plot = heatmap.opts(**fig_kwargs)

    return plot
