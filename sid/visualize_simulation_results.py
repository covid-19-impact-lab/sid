import pandas as pd
from bokeh.plotting import figure, save, show
from utilities.colors import get_colors
from bokeh.models import Column, Div
from bokeh.io import output_file, export_png
import os
from shutil import rmtree
from pandas.api.types import is_categorical


def visualize_simulation_results(
    data_path, background_vars=["age_group", "sector"], outdir_path=None, show_layout=False, title=None
):
    if isinstance(background_vars, str):
        background_vars = [background_vars]
    title = "Visualization of the Simulation Results" if title is None else title
    data = pd.read_pickle(data_path)
    data["symptomatic_among_infectious"] = data["symptoms"].where(data["infectious"])

    plots_and_divs = _create_plots_and_divs(data=data, background_vars=background_vars)

    if outdir_path is not None:
        # empty folders
        if os.path.exists(outdir_path):
            rmtree(outdir_path)
        os.mkdir(outdir_path)
        os.mkdir(outdir_path / "incidences")
        os.mkdir(outdir_path / "r_zeros")
        for var in background_vars:
            os.mkdir(outdir_path / "incidences" / var)

        # export plots as png.
        output_file(outdir_path)
        for element in plots_and_divs:
            name = element.name
            if name.startswith("plot"):
                name = name[5:]
                export_png(element, filename=outdir_path / f"{name}.png")

        # create tex and pdf

    col = Column(*plots_and_divs)

    if outdir_path is not None:
        output_file(f"{outdir_path}/webpage.html")
        save(col, title=title)

    if show_layout is True:
        show(col)


def _create_plots_and_divs(data, background_vars):
    infection_vars = [
        "ever_infected",
        "infectious",
        "symptomatic_among_infectious",
        "needs_icu",
        "dead",
        "immune",
    ]
    style = {"font-size": "150%"}

    # infection rates in the general population
    inf_title = "Infection Related Rates in the Population"
    inf_title = Div(text=inf_title, style=style, name="div_inf_title")
    inf_rates = _plot_infection_rates(
        data=data, infection_vars=infection_vars, colors=get_colors("categorical", 12)
    )
    elements = [inf_title, inf_rates]

    # infection rates in subpopulations
    for groupby_var in background_vars:
        gb_title = f"Infection Related Rates by {_nice_str(groupby_var)}"
        gb_title = Div(text=gb_title, style=style, name=f"div_{groupby_var}_inf_title")
        gb_rates = _plot_rates_by_group(
            data=data, groupby_var=groupby_var, infection_vars=infection_vars,
            colors=get_colors("categorical", 12)
        )
        elements += [gb_title, *gb_rates]

        # r zeros
        r_title = f"Basic Replication Rate Overall and by {_nice_str(groupby_var)}"
        r_title = Div(text=r_title, style=style, name="div_r_title")
        r_zeros = _plot_r_zeros(data=data, groupby_var=groupby_var)
        elements += [r_title, r_zeros]
    return elements


def _plot_infection_rates(data, infection_vars, colors):
    overall_gb = data.groupby("period")
    overall_means = overall_gb[infection_vars].mean()
    title = "Infection Related Rates in the General Population"
    p = _plot_rates(means=overall_means, colors=colors, title=title)
    p.name = "plot_incidences/general"
    return p


def _plot_rates_by_group(data, groupby_var, infection_vars, colors):
    gb = data.groupby(["period", groupby_var])
    plots = []
    if is_categorical(data[groupby_var]):
        ordered = data[groupby_var].cat.ordered
        n_categories = len(data[groupby_var].cat.categories)
    else:
        ordered = False
    str_for_ordered = ["red", "blue", "yellow", "purple", "orange", "green"]
    for i, var in enumerate(infection_vars):
        plot_colors = get_colors(str_for_ordered[i], n_categories) if ordered else colors
        means = gb[var].mean().unstack()
        title = f"{_nice_str(var)} Rates by {_nice_str(groupby_var)}"
        p = _plot_rates(means=means, colors=plot_colors, title=title)
        p.name = f"plot_incidences/{groupby_var}/{var}_rate"
        plots.append(p)
    return plots


def _plot_r_zeros(data, groupby_var=None):
    r_colors = ["black"]
    if is_categorical(data[groupby_var]):
        n_categories = len(data[groupby_var].cat.categories)
        if data[groupby_var].cat.ordered:
            r_colors += get_colors("red", n_categories)
        else:
            r_colors += get_colors("categorical", n_categories)
    else:
        r_colors += get_colors("categorical", 12)

    gb = data.groupby("period")
    overall_r_zeros = gb.apply(_calc_r_zero).to_frame(name="overall")
    gb = data.groupby(["period", groupby_var])
    group_r_zeros = gb.apply(_calc_r_zero).unstack()

    to_plot = pd.merge(
        overall_r_zeros, group_r_zeros, left_index=True, right_index=True
    )
    p = _plot_rates(
        to_plot,
        colors=r_colors,
        title=f"Basic Replication Rate by {_nice_str(groupby_var)}",
    )
    p.name = f"plot_r_zeros/{groupby_var}"
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
