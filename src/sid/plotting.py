import itertools

import holoviews as hv
import pandas as pd
from bokeh.models import HoverTool
from sid.colors import get_colors
from sid.policies import compute_pseudo_effect_sizes_of_policies


DEFAULT_FIGURE_KWARGS = {
    "height": 400,
    "width": 600,
    "line_width": 12,
    "title": "Gantt Chart of Policies",
    "bar_height": 0.8,
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

    gantt_opts = segments.opts(
        color="color",
        alpha="alpha",
        tools=[hover],
        yticks=y_ticks_and_labels,
        **fig_kwargs,
    )

    return gantt_opts


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
