import itertools
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import dask.dataframe as dd
import holoviews as hv
import numpy as np
import pandas as pd
from bokeh.models import HoverTool
from sid.colors import get_colors
from sid.policies import compute_pseudo_effect_sizes_of_policies


DEFAULT_FIGURE_KWARGS = {
    "height": 400,
    "width": 600,
    "line_width": 12,
    "title": "Gantt Chart of Policies",
}


def plot_policy_gantt_chart(
    policies,
    effects=False,
    colors="categorical",
    fig_kwargs=None,
):
    """Plot a Gantt chart of the policies."""
    if fig_kwargs is None:
        fig_kwargs = {}
    fig_kwargs = {**DEFAULT_FIGURE_KWARGS, **fig_kwargs}

    if isinstance(policies, dict):
        df = (
            pd.DataFrame(policies)
            .T.reset_index()
            .rename(columns={"index": "name"})
            .astype({"start": "datetime64", "end": "datetime64"})
            .drop(columns="policy")
        )
    elif isinstance(policies, pd.DataFrame):
        df = policies
    else:
        raise ValueError("'policies' should be either a dict or pandas.DataFrame.")

    if effects:
        effect_kwargs = effects if isinstance(effects, dict) else {}
        effects = compute_pseudo_effect_sizes_of_policies(
            policies=policies, **effect_kwargs
        )
        effects_s = pd.DataFrame(
            [{"policy": name, "effect": effects[name]["mean"]} for name in effects]
        ).set_index("policy")["effect"]
        df = df.merge(effects_s, left_on="name", right_index=True)
        df["alpha"] = (1 - df["effect"] + 0.1) / 1.1
    else:
        df["alpha"] = 1

    df = df.reset_index()
    df = _complete_dates(df)
    df = _add_color_to_gantt_groups(df, colors)
    df = _add_positions(df)

    hv.extension("bokeh", logo=False)

    segments = hv.Segments(
        df,
        [
            hv.Dimension("start", label="Date"),
            hv.Dimension("position", label="Affected contact model"),
            "end",
            "position",
        ],
    )
    y_ticks_and_labels = list(zip(*_create_y_ticks_and_labels(df)))

    tooltips = [("Name", "@name")]
    if effects:
        tooltips.append(("Effect", "@effect"))
    hover = HoverTool(tooltips=tooltips)

    gantt = segments.opts(
        color="color",
        alpha="alpha",
        tools=[hover],
        yticks=y_ticks_and_labels,
        **fig_kwargs,
    )

    return gantt


def _complete_dates(df):
    """Complete dates."""
    for column in ("start", "end"):
        df[column] = pd.to_datetime(df[column])
    df["start"] = df["start"].fillna(df["start"].min())
    df["end"] = df["end"].fillna(df["end"].max())
    return df


def _add_color_to_gantt_groups(df, colors):
    """Add a color for each affected contact model."""
    colors_ = itertools.cycle(get_colors(colors, 4))
    acm_to_color = dict(zip(df["affected_contact_model"].unique(), colors_))
    df["color"] = df["affected_contact_model"].replace(acm_to_color)

    return df


def _add_positions(df):
    """Add positions.

    This functions computes the positions of policies, displayed as segments on the time
    line. For example, if two policies affecting the same contact model have an
    overlapping time windows, the segments are stacked and drawn onto different
    horizontal lines.

    """
    min_position = 0

    def _add_within_group_positions(df):
        """Add within group positions."""
        nonlocal min_position
        position = pd.Series(data=min_position, index=df.index)
        for i in range(1, len(df)):
            start = df.iloc[i]["start"]
            end = df.iloc[i]["end"]
            is_overlapping = (
                (df.iloc[:i]["start"] <= start) & (start <= df.iloc[:i]["end"])
            ) | ((df.iloc[:i]["start"] <= end) & (end <= df.iloc[:i]["end"]))
            if is_overlapping.any():
                possible_positions = set(range(min_position, i + min_position + 1))
                positions_of_overlapping = set(position.iloc[:i][is_overlapping])
                position.iloc[i] = min(possible_positions - positions_of_overlapping)

        min_position = max(position) + 1

        return position

    positions = df.groupby("affected_contact_model", group_keys=False).apply(
        _add_within_group_positions
    )
    df["position_local"] = positions
    df["position"] = df.groupby(
        ["affected_contact_model", "position_local"], sort=True
    ).ngroup()

    return df


def _create_y_ticks_and_labels(df):
    """Create the positions and their related labels for the y axis."""
    pos_per_group = df.groupby("position", as_index=False).first()
    mean_pos_per_group = (
        pos_per_group.groupby("affected_contact_model")["position"].mean().reset_index()
    )
    return mean_pos_per_group["position"], mean_pos_per_group["affected_contact_model"]


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
    "cmap": "YlOrBr",
}


def plot_infection_rates_by_contact_models(
    df_or_time_series: Union[pd.DataFrame, dd.core.DataFrame],
    show_reported_cases: bool = False,
    unit: str = "share",
    fig_kwargs: Optional[Dict[str, Any]] = None,
) -> hv.HeatMap:
    """Plot infection rates by contact models.

    Parameters
    ----------
    df_or_time_series : Union[pandas.DataFrame, dask.dataframe.core.DataFrame]
        The input can be one of the following two.

        1. It is a :class:`dask.dataframe.core.DataFrame` which holds the time series
           from a simulation.
        2. It can be a :class:`pandas.DataFrame` which is created with
           :func:`prepare_data_for_infection_rates_by_contact_models`. It allows to
           compute the data for various simulations with different seeds and use the
           average over all seeds.

    show_reported_cases : bool, optional
        A boolean to select between reported or real cases of infections. Reported cases
        are identified via testing mechanisms.
    unit : str
        The arguments specifies the unit shown in the figure.

        - ``"share"`` means that daily units represent the share of infection caused
          by a contact model among all infections on the same day.

        - ``"population_share"`` means that daily units represent the share of
          infection caused by a contact model among all people on the same day.

        - ``"incidence"`` means that the daily units represent incidence levels per
          100,000 individuals.
    fig_kwargs : Optional[Dict[str, Any]], optional
        Additional keyword arguments which are passed to ``heatmap.opts`` to style the
        plot. The keyword arguments overwrite or extend the default arguments.

    Returns
    -------
    heatmap : hv.HeatMap
        The heatmap object.

    """
    fig_kwargs = (
        DEFAULT_IR_PER_CM_KWARGS
        if fig_kwargs is None
        else {**DEFAULT_IR_PER_CM_KWARGS, **fig_kwargs}
    )

    if _is_data_prepared_for_heatmap(df_or_time_series):
        df = df_or_time_series
    else:
        df = prepare_data_for_infection_rates_by_contact_models(
            df_or_time_series, show_reported_cases, unit
        )

    hv.extension("bokeh", logo=False)

    heatmap = hv.HeatMap(df)
    plot = heatmap.opts(**fig_kwargs)

    return plot


def _is_data_prepared_for_heatmap(df):
    """Is the data prepared for the heatmap plot."""
    return (
        isinstance(df, pd.DataFrame)
        and df.columns.isin(["date", "channel_infected_by_contact", "share"]).all()
        and not df["channel_infected_by_contact"]
        .isin(["not_infected_by_contact"])
        .any()
    )


def prepare_data_for_infection_rates_by_contact_models(
    time_series: dd.core.DataFrame,
    show_reported_cases: bool = False,  # noqa: U100
    unit: str = "share",
) -> pd.DataFrame:
    """Prepare the data for the heatmap plot.

    Parameters
    ----------
    time_series : dask.dataframe.core.DataFrame
        The time series of a simulation.
    show_reported_cases : bool, optional
        A boolean to select between reported or real cases of infections. Reported cases
        are identified via testing mechanisms.
    unit : str
        The arguments specifies the unit shown in the figure.

        - ``"share"`` means that daily units represent the share of infection caused
          by a contact model among all infections on the same day.

        - ``"population_share"`` means that daily units represent the share of
          infection caused by a contact model among all people on the same day.

        - ``"incidence"`` means that the daily units represent incidence levels per
          100,000 individuals.

    Returns
    -------
    time_series : pandas.DataFrame
        The time series with the prepared data for the plot.

    """
    if isinstance(time_series, pd.DataFrame):
        time_series = dd.from_pandas(time_series, npartitions=1)
    elif not isinstance(time_series, dd.core.DataFrame):
        raise ValueError("'time_series' must be either pd.DataFrame or dask.dataframe.")

    if "channel_infected_by_contact" not in time_series:
        raise ValueError(ERROR_MISSING_CHANNEL)

    if show_reported_cases:
        time_series = _adjust_channel_infected_by_contact_to_new_known_cases(
            time_series
        )

    counts = (
        time_series[["date", "channel_infected_by_contact"]]
        .groupby(["date", "channel_infected_by_contact"])
        .size()
        .reset_index()
        .rename(columns={0: "n"})
    )

    if unit == "share":
        out = counts.query(
            "channel_infected_by_contact != 'not_infected_by_contact'"
        ).assign(
            share=lambda x: x["n"]
            / x.groupby("date")["n"].transform("sum", meta=("n", "f8")),
        )

    elif unit == "population_share":
        out = counts.assign(
            share=lambda x: x["n"]
            / x.groupby("date")["n"].transform("sum", meta=("n", "f8")),
        ).query("channel_infected_by_contact != 'not_infected_by_contact'")

    elif unit == "incidence":
        out = counts.query(
            "channel_infected_by_contact != 'not_infected_by_contact'"
        ).assign(share=lambda x: x["n"] * 7 / 100_000)

    else:
        raise ValueError(
            "'unit' should be one of 'share', 'population_share' or 'incidence'"
        )

    out = out.drop(columns="n").compute()

    return out


def _adjust_channel_infected_by_contact_to_new_known_cases(df):
    """Adjust channel of infections by contacts to new known cases.

    Channel of infections are recorded on the date an individual got infected which is
    not the same date an individual is tested positive with a PCR test.

    This function adjusts ``"channel_infected_by_contact"`` such that the infection
    channel is shifted to the date when an individual is tested positive.

    """
    channel_of_infection_by_contact = _find_channel_of_infection_for_individuals(df)
    df = _patch_channel_infected_by_contact(df, channel_of_infection_by_contact)
    return df


def _find_channel_of_infection_for_individuals(df):
    """Find the channel of infected by contact for each individual."""
    df["channel_infected_by_contact"] = df["channel_infected_by_contact"].cat.as_known()

    df["channel_infected_by_contact"] = df[
        "channel_infected_by_contact"
    ].cat.remove_categories(["not_infected_by_contact"])

    df = df.dropna(subset=["channel_infected_by_contact"])

    df = df["channel_infected_by_contact"].compute()

    return df


def _patch_channel_infected_by_contact(df, s):
    """Patch channel of infections by contact to only show channels for known cases."""
    df = df.drop(columns="channel_infected_by_contact")

    df = df.merge(s.to_frame(name="channel_infected_by_contact"), how="left")

    df["channel_infected_by_contact"] = df["channel_infected_by_contact"].mask(
        ~df["new_known_case"], np.nan
    )

    df["channel_infected_by_contact"] = (
        df["channel_infected_by_contact"]
        .cat.add_categories("not_infected_by_contact")
        .fillna("not_infected_by_contact")
    )

    return df
