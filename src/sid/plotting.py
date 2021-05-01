import matplotlib.pyplot as plt
import pandas as pd
from sid.colors import get_colors


def plot_policy_gantt_chart(policies, title=None, bar_height=0.8, alpha=0.5):
    """Plot a Gantt chart of the policies."""
    if isinstance(policies, dict):
        df = pd.DataFrame(policies).T.reset_index().rename(columns={"index": "name"})
    elif isinstance(policies, pd.DataFrame):
        df = policies
    else:
        raise ValueError("'policies' should be either a dict or pandas.DataFrame.")

    df = _complete_dates(df)
    df = _add_color_to_gantt_groups(df)
    df = _add_positions(df)

    _, ax = plt.subplots(figsize=(12, df["position"].max() + 1))
    for _, row in df.iterrows():
        start = pd.Timestamp(row["start"])
        end = pd.Timestamp(row["end"])
        ax.broken_barh(
            xranges=[(start, end - start)],
            yrange=(row["position"] - 0.5 * bar_height, bar_height),
            edgecolors=row["color"],
            facecolors=row["color"],
            alpha=alpha,
            label=row["name"],
        )

        ax.text(
            start + (end - start) * 0.1,
            row["position"] + 0.375 - 0.5 * bar_height,
            row["name"],
        )

    positions, labels = _create_y_ticks_and_labels(df)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)

    if title is not None:
        ax.set_title(title)

    return ax


def _complete_dates(df):
    for column in ("start", "end"):
        df[column] = pd.to_datetime(df[column])
    df["start"] = df["start"].fillna(df["start"].min())
    df["end"] = df["end"].fillna(df["end"].max())
    return df


def _add_color_to_gantt_groups(df):
    """Add a color for each affected contact model."""
    n_colors = len(df["affected_contact_model"].unique())
    colors = get_colors("categorical", n_colors)
    acm_to_color = dict(zip(df["affected_contact_model"].unique(), colors))
    df["color"] = df["affected_contact_model"].replace(acm_to_color)

    return df


def _add_positions(df):
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
    df["position"] = positions

    return df


def _create_y_ticks_and_labels(df):
    pos_per_group = df.groupby("position", as_index=False).first()
    mean_pos_per_group = (
        pos_per_group.groupby("affected_contact_model")["position"].mean().reset_index()
    )
    return mean_pos_per_group["position"], mean_pos_per_group["affected_contact_model"]
