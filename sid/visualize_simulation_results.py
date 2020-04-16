import numpy as np
import pandas as pd
from bokeh.plotting import figure, save, show
from utilities.colors import get_colors
from bokeh.models import Column, Div
from bokeh.io import output_file


def visualize_simulation_results(
    data_path, groupby_var="age_group", outdir_path=None, show=False, title=None
):
    title = "Visualization of the Simulation Results" if title is None else title
    data = pd.read_pickle(data_path)
    data["symptomatic_among_infectious"] = data["symptoms"].where(data["infectious"])
    colors = get_colors("categorical", 12)
    infection_vars = [
        "ever_infected",
        "infectious",
        "symptomatic_among_infectious",
        "needs_icu",
        "dead",
        "immune",
    ]

    inf_rates = _plot_infection_rates(
        data=data, infection_vars=infection_vars, colors=colors
    )
    gb_rates = _plot_rates_by_group(
        data=data, groupby_var=groupby_var, infection_vars=infection_vars, colors=colors
    )
    r_zeros = _plot_r_zeros(data=data, groupby_var=groupby_var, colors=colors)

    # export plots as png.
    # create tex and pdf

    style = {"font-size": "150%"}  # , "color": "#808080"}
    inf_title = "Infection Related Rates in the Population"
    inf_title = Div(text=inf_title, style=style)
    gb_title = f"Infection Related Rates by {_nice_str(groupby_var)}"
    gb_title = Div(text=gb_title, style=style)
    r_title = f"Basic Replication Rate Overall and by {_nice_str(groupby_var)}"
    r_title = Div(text=r_title, style=style)

    col = Column(inf_title, inf_rates, gb_title, *gb_rates, r_title, r_zeros)

    if outdir_path is not None:
        output_file(f"{outdir_path}/webpage.html")
        save(col, title=title)

    if show is True:
        show(col)


def _plot_infection_rates(data, infection_vars, colors):
    overall_gb = data.groupby("period")
    overall_means = overall_gb[infection_vars].mean()
    title = "Infection Related Rates in the General Population"
    p = _plot_rates(means=overall_means, colors=colors, title=title)
    return p


def _plot_rates_by_group(data, groupby_var, infection_vars, colors):
    gb = data.groupby(["period", groupby_var])
    plots = []
    for var in infection_vars:
        means = gb[var].mean().unstack()
        title = f"{_nice_str(var)} Rates by {_nice_str(groupby_var)}"
        plots.append(_plot_rates(means=means, colors=colors, title=title))
    return plots


def _plot_r_zeros(data, colors, groupby_var=None):
    gb = data.groupby("period")
    overall_r_zeros = gb.apply(_calc_r_zero).to_frame(name="overall")
    gb = data.groupby(["period", groupby_var])
    group_r_zeros = gb.apply(_calc_r_zero).unstack()
    to_plot = pd.merge(
        overall_r_zeros, group_r_zeros, left_index=True, right_index=True
    )
    p = _plot_rates(
        to_plot,
        colors=colors,
        title=f"Basic Replication Rate by {_nice_str(groupby_var)}",
    )
    return p


def _plot_rates(means, title, colors):
    """Plot the rates over time.

    Args:
        means (DataFrame):
            the index are the x values, the values the y values.
            every column is plotted as a separate line.
        colors (list): full color palette
        title (str): plot title.

    """
    p = figure(tools=[], plot_height=300, plot_width=600, title=title,)
    for var, color in zip(means, colors[: len(means.columns)]):
        rates = means[var]
        p.line(
            x=rates.index, y=rates, line_width=2.0, legend_label=var, line_color=color,
        )
    if means.columns.name is not None:
        p.legend.title = _nice_str(means.columns.name)
    p = _style(p)
    return p


def _style(p):
    gray = "#808080"
    p.outline_line_color = None
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.axis.minor_tick_line_color = None
    p.axis.axis_line_color = gray
    p.axis.major_label_text_color = gray
    p.axis.major_tick_line_color = gray
    p.legend.border_line_alpha = 0.0
    p.legend.background_fill_alpha = 0.0
    p.legend.location = "top_left"
    p.legend.title_text_font_style = "normal"
    if len(p.legend.items) < 2:
        p.legend.visible = False
    # p.legend.label_text_color = gray
    # p.title.text_color= gray
    return p


def _calc_r_zero(df, n_periods=1):
    pop_of_interest = df[df["cd_infectious_false"] >= -n_periods]
    r_zero = pop_of_interest["infection_counter"].mean()
    return r_zero


def _nice_str(s):
    return s.replace("_", " ").title()
