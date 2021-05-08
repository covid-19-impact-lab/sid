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


def plot_infection_rates_by_contact_models(df_or_time_series, fig_kwargs=None):
    """Plot infection rates by contact models."""
    fig_kwargs = (
        DEFAULT_IR_PER_CM_KWARGS
        if fig_kwargs is None
        else {**DEFAULT_IR_PER_CM_KWARGS, **fig_kwargs}
    )

    if _is_data_prepared_for_heatmap(df_or_time_series):
        df = df_or_time_series
    else:
        df = prepare_data_for_infection_rates_by_contact_models(df_or_time_series)

    hv.extension("bokeh")

    heatmap = hv.HeatMap(df)
    plot = heatmap.opts(**fig_kwargs)

    return plot


def _is_data_prepared_for_heatmap(df):
    """Is the data prepared for the heatmap plot."""
    return (
        isinstance(df, pd.DataFrame)
        and df.columns.isin(["date", "channel_infected_by_contact", "share"]).all()
        and not df["channel_infected_by_contact"].isin("not_infected_by_contact").any()
    )


def prepare_data_for_infection_rates_by_contact_models(time_series):
    """Prepare the data for the heatmap plot."""
    if isinstance(time_series, pd.DataFrame):
        time_series = dd.from_pandas(time_series, npartitions=1)
    elif not isinstance(time_series, dd.core.DataFrame):
        raise ValueError("'time_series' must be either pd.DataFrame or dask.dataframe.")

    if "channel_infected_by_contact" not in time_series:
        raise ValueError(ERROR_MISSING_CHANNEL)

    time_series = (
        time_series[["date", "channel_infected_by_contact"]]
        .groupby(["date", "channel_infected_by_contact"])
        .size()
        .reset_index()
        .rename(columns={0: "n"})
        .assign(
            share=lambda x: x["n"]
            / x.groupby("date")["n"].transform("sum", meta=("n", "f8")),
        )
        .drop(columns="n")
        .query("channel_infected_by_contact != 'not_infected_by_contact'")
    )
    if isinstance(time_series, dd.core.DataFrame):
        time_series = time_series.compute()

    return time_series
