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

from sid.shared import calculate_r_effective
from sid.shared import calculate_r_zero


def visualize_simulation_results(
    data_paths, outdir_path, infection_vars, background_vars, window_length,
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

    rates = _create_statistics(
        data_paths, infection_vars, background_vars, window_length
    )
    for bg_var in ["general"] + background_vars:
        if bg_var == "general":
            title = "Rates in the General Population"
        else:
            title = f"Rates According to {_nice_str(bg_var)}"  # noqa
        rate_plots = _create_rate_plots(rates, bg_var, colors, title)
        _export_plots_and_layout(
            title=Div(text=title, style={"font-size": "150%"}),
            plots=rate_plots,
            outdir_path=outdir_path / bg_var,
        )


def _create_statistics(data_paths, infection_vars, background_vars, window_length):
    """Calculate the infection rates and reproduction numbers for each date.

    Args:
        data_paths (list): list of paths to the pickled simulation results
        infection_vars (list): list of infection rates to plot
        background_vars (list): list of background variables by whose value to group
            the results. Have to be present in all simulation results.
        window_length (int): how many dates to use for the reproduction numbers.

    Returns:
        rates (pd.DataFrame): DataFrame with the dates as index.
            The columns are a MultiIndex with four levels: The outermost is the
            "rate". The next is the "bg_var"
            ("general" for the overall rate), "bg_value" and last "data_id".

    """
    vars_for_r_zero = ["immune", "infection_counter", "cd_infectious_false"]
    keep_vars = sorted(set(infection_vars + background_vars + vars_for_r_zero))
    name_to_means = {}
    for path in data_paths:
        data = pd.read_pickle(path)[keep_vars]
        gb = data.groupby("date")

        overall = gb.mean()[infection_vars]
        overall["r_zero"] = gb.apply(calculate_r_zero, window_length)
        overall["r_effective"] = gb.apply(calculate_r_effective, window_length)

        # add column levels for later
        overall.columns.name = "rate"
        overall = _prepend_column_level(overall, "general", "bg_value")
        overall = _prepend_column_level(overall, "general", "bg_var")

        single_df_rates = [overall]

        for bg_var in background_vars:
            gb = data.groupby([bg_var, "date"])
            infection_rates = gb.mean()[infection_vars].unstack(level=0)
            r_zeros = gb.apply(calculate_r_zero, window_length).unstack(level=0)
            r_zeros = _prepend_column_level(r_zeros, "r_zero", "rate")
            r_eff = gb.apply(calculate_r_effective, window_length).unstack(level=0)
            r_eff = _prepend_column_level(r_eff, "r_effective", "rate")

            rates_by_group = pd.concat([infection_rates, r_zeros, r_eff], axis=1)
            rates_by_group.columns.names = ["rate", "bg_value"]
            rates_by_group = _prepend_column_level(rates_by_group, bg_var, "bg_var")
            rates_by_group = rates_by_group.swaplevel("rate", "bg_value", axis=1)
            single_df_rates.append(rates_by_group)

        name_to_means[path.stem] = pd.concat(single_df_rates, axis=1)

    infection_rates = pd.concat(name_to_means, axis=1, names=["data_id"])
    order = ["rate", "bg_var", "bg_value", "data_id"]
    infection_rates = infection_rates.reorder_levels(order=order, axis=1)
    return infection_rates


def _create_rate_plots(rates, bg_var, colors, title):
    """Plot all rates for a single background variable

    Args:
        rates (pd.DataFrame): DataFrame with the dates as index.
            The columns are a MultiIndex with four levels: The outermost is the
            "Infection Variable". The next is the "bg_var"
            ("general" for the overall rate), "bg_value" and last "data_id".
        bg_var (str): background variable. Value that the second index level can take.

    Returns:
        plots (list): list of bokeh plots.

    """
    vars_to_plot = rates.columns.levels[0]
    plots = []
    for var, color in zip(vars_to_plot, colors):
        full_range_vars = ["ever_infected", "immune", "symptomatic_among_infectious"]
        y_range = (0, 1) if var in full_range_vars else None
        background_values = rates[var][bg_var].columns.unique().levels[0]
        for bg_val in background_values:
            to_plot = rates[var][bg_var][bg_val]
            plot_title = f"{_nice_str(var)} {title}"
            if bg_val != "general":
                plot_title += f": {bg_val}"  # noqa
            p = _plot_rates(to_plot, plot_title, color, y_range)
            if bg_var == "general":
                p.name = var
            else:
                p.name = f"{var}_{bg_val.replace(' ', '')}"
            plots.append(p)
    return plots


def _plot_rates(rates, title, color, y_range):
    """Plot the rates over time.

    Args:
        rates (DataFrame): the index are the x values, the values the y values.
            Every column is plotted as a separate line.
        color (str): color.
        title (str): plot title.
        y_range (tuple or None): range of the y axis.

    Returns:
        p (bokeh figure)

    """
    xs = rates.index
    p = figure(
        tools=[],
        plot_height=400,
        plot_width=800,
        title=title,
        y_range=y_range,
        x_axis_type="datetime",
    )

    for var in rates:
        p.line(x=xs, y=rates[var], line_width=1, line_color=color, alpha=0.3)

    p.line(x=xs, y=rates.median(axis=1), alpha=1, line_width=2.75, line_color=color)

    q5 = rates.apply(np.nanpercentile, q=5, axis=1)
    q95 = rates.apply(np.nanpercentile, q=95, axis=1)
    p.varea(x=xs, y1=q95, y2=q5, alpha=0.2, color=color)

    p = _style(p)
    return p


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
    for var in ["general"] + background_vars:
        os.mkdir(outdir_path / var)


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


def _prepend_column_level(df, key, name):
    prepended = pd.concat([df], keys=[key], names=[name], axis=1)
    return prepended
