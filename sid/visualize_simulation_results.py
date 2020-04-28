import os
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
from bokeh.io import export_png
from bokeh.io import output_file
from bokeh.models import Column
from bokeh.models import Div
from bokeh.plotting import figure
from bokeh.plotting import save
from utilities.colors import get_colors

# from sid.shared import calculate_r_zero


def visualize_simulation_results(
    data_paths, outdir_path, infection_vars, background_vars
):
    """Visualize the results one or more simulation results.

    Args:
        data_paths (list): list of paths to the pickled simulation results
        outdir_path (path): path to the folder where to save the results.
            Careful, all contents are removed when the function is called.
        infection_vars (list): list of infection rates to plot
        background_vars (list): list of background variables by whose value to group
            the results. Have to be present in all simulation results.

    """
    data_paths, outdir_path, background_vars, colors = _input_processing(
        data_paths, outdir_path, background_vars
    )

    _create_folders(outdir_path, background_vars)

    infection_rate_data = _calculate_infection_rates(
        data_paths, infection_vars, background_vars
    )

    # plot all rates and save them.
    for bg_var in ["general"] + background_vars:
        infection_plots = _create_rate_plots(
            rates=infection_rate_data,
            colors=colors,
            bg_var=bg_var,
            title="Rates in the General Population",
        )
        _export_plots_and_layout(
            title=Div(
                text="Rates in the General Population", style={"font-size": "150%"}
            ),
            plots=infection_plots,
            outdir_path=outdir_path / "incidences",
        )

    # plot replication rates


def _calculate_infection_rates(data_paths, infection_vars, background_vars):
    """Calculate the infection rates.

    Args:
        data_paths (list): list of paths to the pickled simulation results
        infection_vars (list): list of infection rates to plot
        background_vars (list): list of background variables by whose value to group
            the results. Have to be present in all simulation results.

    Returns:
        infection_rates (pd.DataFrame): DataFrame with the periods as index.
            The columns are a MultiIndex with four levels: The outermost is the
            "Infection Variable". The next is the "Background Variable"
            ("general" for the overall rate), "Background Value" and last "Data Name".

    """
    name_to_means = {}
    for path in data_paths:
        data = pd.read_pickle(path)[infection_vars + background_vars]

        overall_means = data.groupby("period")[infection_vars].mean()
        # make columns a multiindex that fits to the background variable datasets
        overall_means.columns.name = "Infection Variable"
        overall_means = pd.concat(
            [overall_means], keys=["general"], names=["Background Value"], axis=1
        )
        overall_means = pd.concat(
            [overall_means], keys=["general"], names=["Background Variable"], axis=1
        )
        single_df_means = [overall_means]

        for bg_var in background_vars:
            gb = data.groupby([bg_var, "period"])[infection_vars]
            period_as_index = gb.mean().unstack(level=0)
            # adjust multiindex columns
            period_as_index.columns.names = ["Infection Variable", "Background Value"]
            right_columns = pd.concat(
                [period_as_index], keys=[bg_var], names=["Background Variable"], axis=1
            )
            right_columns = right_columns.swaplevel(
                "Infection Variable", "Background Value", axis=1
            )
            single_df_means.append(right_columns)

        means = pd.concat(single_df_means, axis=1)
        name_to_means[path.stem] = means

    infection_rates = pd.concat(name_to_means, axis=1, names=["Data Name"])
    order = [
        "Infection Variable",
        "Background Variable",
        "Background Value",
        "Data Name",
    ]
    infection_rates = infection_rates.reorder_levels(order=order, axis=1)
    return infection_rates


def _create_rate_plots(rates, bg_var, colors, title):
    """Plot all rates for a single background variable

    Args:
        rates (pd.DataFrame): DataFrame with the periods as index.
            The columns are a MultiIndex with four levels: The outermost is the
            "Infection Variable". The next is the "Background Variable"
            ("general" for the overall rate), "Background Value" and last "Data Name".
        bg_var (str): background variable. Value that the second index level can take.

    Returns:
        plots (list): list of bokeh plots.

    """
    infection_vars = rates.columns.levels[0]
    plots = []
    for var, color in zip(infection_vars, colors):
        title = f"{_nice_str(var)} {title}"  # noqa
        full_range_vars = ["ever_infected", "immune", "symptomatic_among_infectious"]
        y_range = (0, 1) if var in full_range_vars else None
        background_values = rates[var][bg_var].columns.unique().levels[0]
        for bg_val in background_values:
            to_plot = rates[var][bg_var][bg_val]
            p = _plot_rates(to_plot, title, color, y_range)
            if bg_var == "general":
                p.name = var
            else:
                p.name = f"{bg_var}/{var}_{bg_val}"
            plots.append(p)
    return plots


def _export_plots_and_layout(title, plots, outdir_path):
    """Save all plots as png and the layout as html.

    Args:
        title (bokeh.Div): title element.
        plots (list): list of bokeh plots
        outdir_path (pathlib.Path): base path to which to append the plot name to build
            the path where to save each plot.

    """
    for p in plots:
        outpath = outdir_path / f"{p.name}.png"
        output_file(outpath)
        export_png(p, filename=outpath)
    output_file(outdir_path / "overview.html")
    save(Column(title, *plots))


def _plot_rates(means, title, color, y_range):
    """Plot the rates over time.

    Args:
        means (DataFrame): the index are the x values, the values the y values.
            Every column is plotted as a separate line.
        color (str): color.
        title (str): plot title.
        y_range (tuple or None): range of the y axis.

    Returns:
        p (bokeh figure)

    """
    xs = means.index
    p = figure(tools=[], plot_height=400, plot_width=800, title=title, y_range=y_range)

    for var in means:
        p.line(x=xs, y=means[var], line_width=1, line_color=color, alpha=0.3)

    p.line(x=xs, y=means.median(axis=1), alpha=1, line_width=2.75, line_color=color)

    q5 = means.apply(np.nanpercentile, q=5, axis=1)
    q95 = means.apply(np.nanpercentile, q=95, axis=1)
    p.varea(x=xs, y1=q95, y2=q5, alpha=0.2, color=color)

    p = _style(p)
    return p


def _input_processing(data_paths, outdir_path, background_vars):
    colors = get_colors("categorical", 12)
    if isinstance(background_vars, str):
        background_vars = [background_vars]
    outdir_path = Path(outdir_path)
    data_paths = [data_paths] if isinstance(data_paths, (str, Path)) else data_paths
    return data_paths, outdir_path, background_vars, colors


def _create_folders(outdir_path, background_vars):
    if os.path.exists(outdir_path):
        rmtree(outdir_path)
    os.mkdir(outdir_path)
    os.mkdir(outdir_path / "incidences")
    os.mkdir(outdir_path / "r_zeros")
    for var in background_vars:
        os.mkdir(outdir_path / "incidences" / var)


def _style(p):
    gray = "#808080"
    p.outline_line_color = None
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.axis.minor_tick_line_color = None
    p.axis.axis_line_color = gray
    p.axis.major_label_text_color = gray
    p.axis.major_tick_line_color = gray
    return p


def _nice_str(s):
    return s.replace("_", " ").title()
