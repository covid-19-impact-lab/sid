import os
from pathlib import Path
from shutil import rmtree

import bokeh.plotting
import numpy as np
import pandas as pd
from bokeh.io import export_png
from bokeh.io import output_file
from bokeh.models import Column
from bokeh.models import Div
from bokeh.models import Range1d
from bokeh.plotting import figure
from bokeh.plotting import save
from pandas.api.types import is_categorical
from utilities.colors import get_colors

from sid.shared import calculate_r_zero


def visualize_simulation_results(
    data_paths, outdir_path, background_vars=None, model_name=None, colors=None
):
    """Visualize the results one or more simulation results.

    Args:
        data_paths (list): list of paths to the pickled simulation results
        outdir_path (path): path to the folder where to save the results.
            Careful, all contents are removed when the function is called.
        background_vars (list): list of background variables by whose value to group
            the results. Have to be present in all simulation results.
        model_name (str): name of the model.

    """
    colors = get_colors("categorical", 12) if colors is None else colors
    model_name = "" if model_name is None else model_name
    data_paths = [data_paths] if isinstance(data_paths, (str, Path)) else data_paths
    outdir_path, background_vars = _input_processing(outdir_path, background_vars)

    data_dict = {}
    for path in data_paths:
        data = pd.read_pickle(path)
        data["symptomatic_among_infectious"] = data["symptoms"].where(
            data["infectious"]
        )
        data_dict[path.name.replace(".pkl", "")] = data.sort_index()

    # create folders
    if os.path.exists(outdir_path):
        rmtree(outdir_path)
    os.mkdir(outdir_path)
    os.mkdir(outdir_path / "incidences")
    os.mkdir(outdir_path / "r_zeros")
    for var in ["general"] + background_vars:
        os.mkdir(outdir_path / "incidences" / var)

    _create_plots_and_titles(
        data_dict=data_dict,
        background_vars=background_vars,
        colors=colors,
        outdir_path=outdir_path,
    )


def _create_plots_and_titles(data_dict, background_vars, colors, outdir_path):
    """Create a list of titles and plots showing the results of one or more datasets.

    Args:
        data_dict (dict): mapping from string to DataFrame with the simulation results.
        background_vars (list): list of background variables by whose value to group
            the results. Have to be present in all simulation results.

    Returns:
        elements (list): list of bokeh elements that can be passed to a bokeh layout.

    """
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
        data_dict=data_dict,
        infection_vars=infection_vars,
        colors=colors,
        outdir_path=outdir_path,
    )
    elements = [inf_title, *inf_rates]

    # infection rates in subpopulations
    for groupby_var in background_vars:
        gb_title = f"Infection Related Rates by {_nice_str(groupby_var)}"
        gb_title = Div(text=gb_title, style=style, name=f"div_{groupby_var}_inf_title")
        gb_rates = _plot_rates_by_group(
            data_dict=data_dict,
            groupby_var=groupby_var,
            infection_vars=infection_vars,
            colors=get_colors("categorical", 12),
            outdir_path=outdir_path,
        )
        elements += [gb_title, *gb_rates]
    #     # r zeros
    #     r_title = f"Basic Replication Rate Overall and by {_nice_str(groupby_var)}"
    #     r_title = Div(text=r_title, style=style, name="div_r_title")
    #     r_zeros = _plot_r_zeros(data=data, groupby_var=groupby_var)
    #     elements += [r_title, r_zeros]


def _plot_infection_rates(data_dict, infection_vars, colors, outdir_path, legend=False):
    """Create a plot for every variable with the respective rates in the data.

    Args:
        data_dict (dict): mapping from string to DataFrame with the simulation results.
        infection_vars (list): list of variables whose rates to plot over time.
            These have to be present in all simulation results.
        colors (list): colors
        legend (bool): whether to show a legend or not.

    """
    plots = []
    outpath = outdir_path / "incidences" / "general"
    for i, inf_var in enumerate(infection_vars):
        inf_data = pd.DataFrame()
        for name, data in data_dict.items():
            inf_data[name] = data[inf_var]
        overall_gb = inf_data.groupby("period")
        overall_means = overall_gb.mean()

        if overall_means.min().min() < overall_means.max().max():
            title = f"{_nice_str(inf_var)} Rates in the General Population"
            p = _plot_rates(means=overall_means, colors=colors[i], title=title)
            # confidence band
            q5 = overall_means.apply(lambda x: np.percentile(a=x, q=5), axis=1)
            q95 = overall_means.apply(lambda x: np.percentile(a=x, q=95), axis=1)
            p.varea(x=q5.index, y1=q95, y2=q5, alpha=0.2, color=colors[i])

            p.name = f"plot_incidences/general/{inf_var}"
            p.legend.visible = legend

            output_file(outpath)
            export_png(p, filename=outpath / f"{inf_var}.png")

            plots.append(p)
    col = Column(*plots)
    save(col, filename="overview.html")
    bokeh.plotting.reset_output()
    return plots


def _plot_rates_by_group(data_dict, groupby_var, infection_vars, colors, outdir_path):
    """
    """
    plots = []
    sorted_keys = sorted(data_dict.keys())
    sample_data = data_dict[sorted_keys[0]]
    for key in sorted_keys[1:]:
        assert (sample_data[groupby_var] == data_dict[key][groupby_var]).all()

    categories = sample_data[groupby_var].unique()

    for var in infection_vars:
        inf_data = sample_data[[groupby_var]].copy()
        for name, data in data_dict.items():
            inf_data[name] = data[var]

        var_div = Div(
            text=_nice_str(var), style={"font-size": "125%"}, name=f"div_gb_{var}_title"
        )
        plots.append(var_div)

        gb = inf_data.groupby([groupby_var, "period"])
        means = gb.mean().fillna(0)
        min_ = means.min().min()
        max_ = means.max().max()
        if min_ < max_:
            y_range = Range1d(min_, max_)
            # one plot for every group
            for i, cat in enumerate(categories):
                title = "{} Rates by {}: {}".format(
                    _nice_str(var), _nice_str(groupby_var), _nice_str(cat)
                )
                p = _plot_rates(
                    means=means.loc[cat], colors=colors[i], title=title, y_range=y_range
                )

                # confidence band
                q5 = means.loc[cat].apply(lambda x: np.percentile(a=x, q=5), axis=1)
                q95 = means.loc[cat].apply(lambda x: np.percentile(a=x, q=95), axis=1)
                p.varea(x=q5.index, y1=q95, y2=q5, alpha=0.2, color=colors[i])
                p.name = f"plot_incidences/{groupby_var}/{var}_rate_{cat}"
                p.legend.visible = False
                outpath = outdir_path / "incidences" / groupby_var
                output_file(outpath)
                export_png(p, filename=outpath / f"{var}_rate_{cat}.png")
                plots.append(p)
    return plots


def _plot_rates(means, title, colors, y_range=None):
    """Plot the rates over time.

    Args:
        means (DataFrame):
            the index are the x values, the values the y values.
            every column is plotted as a separate line.
        colors (list): full color palette
        title (str): plot title.

    Returns:
        p (bokeh figure)

    """
    if isinstance(colors, str):
        colors = [colors] * len(means.columns)
    else:
        colors = colors[: len(means.columns)]
    p = figure(tools=[], plot_height=300, plot_width=600, title=title, y_range=y_range)
    for var, color in zip(means, colors):
        rates = means[var]
        p.line(
            x=rates.index,
            y=rates,
            line_width=2.0,
            legend_label=var,
            line_color=color,
            alpha=0.5,
        )
    if means.columns.name is not None:
        p.legend.title = _nice_str(means.columns.name)
    p = _style(p)
    return p


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
    overall_r_zeros = gb.apply(calculate_r_zero).to_frame(name="overall")
    gb = data.groupby(["period", groupby_var])
    group_r_zeros = gb.apply(calculate_r_zero).unstack()

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


def _nice_str(s):
    return s.replace("_", " ").title()


def _input_processing(outdir_path, background_vars):
    if background_vars is None:
        background_vars = ["age_group"]
    elif isinstance(background_vars, str):
        background_vars = [background_vars]

    outdir_path = Path(outdir_path)
    # clean up folder
    if os.path.exists(outdir_path):
        rmtree(outdir_path)
    os.mkdir(outdir_path)

    return outdir_path, background_vars


def _style(p):
    gray = "#808080"
    p.outline_line_color = None
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.axis.minor_tick_line_color = None
    p.axis.axis_line_color = gray
    p.axis.major_label_text_color = gray
    p.axis.major_tick_line_color = gray

    if p.legend is not None:
        p.legend.border_line_alpha = 0.0
        p.legend.background_fill_alpha = 0.0
        p.legend.location = "top_left"
        p.legend.title_text_font_style = "normal"
        if len(p.legend.items) < 2:
            p.legend.visible = False
    # p.legend.label_text_color = gray
    # p.title.text_color= gray
    return p
