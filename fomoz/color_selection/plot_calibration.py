"""
Data visualization module.

This module provides functions for data visualization.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

from itertools import combinations
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
from scipy.interpolate import BSpline

from .calibrate_linear import CalibrateLinear
from .select_object import SelectObject
from .tools import (
    check_ax_limits,
    check_plot_type,
    compute_metrics,
    normalize_plot_style,
    plot_distribution_contours,
    plot_distribution_scatter,
)


class PlotCalibration:
    """Plot the outputs of calibrated color selections."""

    def __init__(
        self,
        calibrations: Union[List, Tuple, object],
        labels: Optional[Union[List[str], str]] = None,
    ):
        r"""
        Initialize a PlotCalibration object for visualizing calibration results.

        This class provides methods to plot performance metrics and results
        from calibrated color selection algorithms.

        Parameters
        ----------
        calibrations : CalibrateLinear, CalibrateGrid, or list/tuple thereof
            One or more calibration objects to plot.
        labels : list or str, optional
            If not None, it defines the legend labels for each calibration. Default is None.

        Examples
        --------
        >>> # Single calibration object
        >>> plotter = PlotCalibration(calib_linear)
        >>>
        >>> # Multiple calibration objects
        >>> plotter = PlotCalibration([calib_linear, calib_grid])
        """
        if not isinstance(calibrations, (list, tuple)):
            calibrations = [calibrations]
        if len(calibrations) == 0:
            raise ValueError(
                "calibrations must contain at least one calibration object"
            )

        self.calibrations = calibrations
        self.n_calibrations = len(calibrations)

        if isinstance(labels, (list, tuple)):
            if len(labels) != self.n_calibrations:
                raise ValueError(
                    f"Number of labels ({len(labels)}) must match "
                    f"number of calibrations ({self.n_calibrations})."
                )
            self.labels = labels
        elif labels is None:
            self.labels = [labels] * self.n_calibrations
        else:
            raise TypeError("labels can be either None, list, or tuple.")

    def plot_metric(
        self,
        metric,
        ax,
        show_legend=False,
        linestyle="-",
        color="k",
        alpha=0.3,
        interp_points=None,
    ):
        """
        Plot a metric of the optimized color cuts.

        Given an input axis, plot either completeness, purity, or failure rate.
        If bootstrap errors were computed, the 68% confidence interval is shown.

        Parameters
        ----------
        metric : str
            The metric symbol. Possibilities are: "C" (completeness),
            "P" (purity), "F" (failure rate).
        ax : matplotlib.axes.Axes
            Axis object.
        show_legend : bool, optional
            If True, show the legend label. Default is False.
        linestyle : list or str, optional
            Line style. Default is "-".
        color : list or str, optional
            Line color. Default is "k".
        alpha : float, optional
            alpha parameter of the 68% confidence interval. Default is 0.3.
        interp_points : array-like, optional
            Limiting redshift points. If not None, derive the metrics at these points
            using interpolations. Valid only for CalibrateLinear instances. Default is None.
        """
        if metric not in ["C", "P", "F"]:
            raise ValueError(
                "The metric parameter can only take the following values: 'C', 'P', 'F'."
            )

        linestyle = normalize_plot_style(linestyle, "linestyle", self.n_calibrations)
        color = normalize_plot_style(color, "color", self.n_calibrations)

        for i, calibration in enumerate(self.calibrations):
            z_lim = calibration.td.z_lim
            if metric == "C":
                y = calibration.C
                y_std = calibration.C_std
            elif metric == "P":
                y = calibration.P
                y_std = calibration.P_std
            else:
                y = calibration.F
                y_std = calibration.F_std

            ax.plot(
                z_lim,
                y * 100.0,
                color=color[i],
                linestyle=linestyle[i],
                label=self.labels[i] if show_legend else None,
            )

            if not np.any(np.isnan(y_std)):  # Add error bands if available
                ax.fill_between(
                    z_lim,
                    (y - y_std) * 100.0,
                    (y + y_std) * 100.0,
                    color=color[i],
                    alpha=0.3,
                )

            if interp_points is not None and isinstance(calibration, CalibrateLinear):
                sel = SelectObject()
                sel.load_parameters(calibration)

                dataset = calibration.dataset
                sel.process(interp_points, dataset)

                y = []
                for j, z_l in enumerate(interp_points):
                    target_mask = calibration.td.get_target_mask(dataset, [z_l])
                    TP, FP, FN, TN = compute_metrics(
                        np.array(sel.mask[j]), np.array(target_mask[0])
                    )
                    if metric == "C":
                        y.append(TP / (TP + FN) * 100.0 if TP + FN > 0 else 0.0)
                    elif metric == "P":
                        y.append(TP / (TP + FP) * 100.0 if TP + FP > 0 else 0.0)
                    else:
                        y.append(FP / (FP + TN) * 100.0 if FP + TN > 0 else 0.0)

                ax.plot(
                    interp_points,
                    y,
                    label=(
                        self.labels[i] + " - interpolation"
                        if show_legend and self.labels[i] is not None
                        else None
                    ),
                )

    def plot_all_metrics(
        self,
        figsize=(15, 4.5),
        wspace=0.35,
        loclegend="upper right",
        show_legend=False,
        legend_fontsize=13,
        labelsize=13,
        xlabel=r"$z_{\rm l}$",
        math_ylabels=False,
        fontsize=15,
        linestyle="-",
        color="k",
        show_axes_grid=False,
        alpha=0.3,
        interp_points=None,
        output_file_path=None,
        dpi=300,
    ):
        r"""
        Plot the performance metrics of the optimized color cuts.

        This method produces a three-panel figure showing:
        1. Completeness as a function of redshift threshold.
        2. Purity as a function of redshift threshold.
        3. Failure rate as a function of redshift threshold.

        If bootstrap errors were computed, the 68% confidence intervals are shown.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default is (15, 4.5).
        wspace : float, optional
            Space between the subplots. Default is 0.35.
        loclegend : str, optional
            Location of the legend in the first panel. Default is "upper right".
        show_legend : bool, optional
            If True, show the legend. Default is False.
        legend_fontsize : float, optional
            Size of the legend entries. Default is 15.
        labelsize : float, optional
            Size of the tick labels. Default is 13.
        xlabel : str, optional
            Label of the x axis. Default is r"$z_{\rm l}$".
        math_ylabels : bool, optional
            If True, y-axis labels are math symbols. Default is False.
        fontsize : float, optional
            Size of the axis labels. Default is 15.
        linestyle : list or str, optional
            Line style. Default is "-".
        color : list or str, optional
            Line color. Default is "k".
        show_axes_grid : bool, optional
            If True, show the axes tick grid. Default is False.
        alpha : float, optional
            alpha parameter of the 68% confidence interval. Default is 0.3.
        interp_points : array-like, optional
            Limiting redshift points. If not None, derive the metrics at these points
            using interpolations. Valid only for CalibrateLinear instances. Default is None.
        output_file_path : str, optional
            Concatenation of the output file path and name.
            If None, the plot is not saved to file. Default is None.
        dpi : int, optional
            dpi parameter of `plt.savefig`.
        """
        # Set up the plot
        fig, ax = plt.subplots(
            1, 3, figsize=figsize, sharex="row", gridspec_kw={"wspace": wspace}
        )
        for i in range(len(ax)):
            ax[i].tick_params(direction="in", which="both", labelsize=labelsize)
            if show_axes_grid:
                ax[i].grid()
            ax[i].set_xlabel(xlabel, fontsize=fontsize)

        if math_ylabels:
            ax[0].set_ylabel(r"$\mathcal{C}$ [%]", fontsize=fontsize)
            ax[1].set_ylabel(r"$\mathcal{P}$ [%]", fontsize=fontsize)
            ax[2].set_ylabel(r"$\mathcal{F}$ [%]", fontsize=fontsize)
        else:
            ax[0].set_ylabel(r"Completeness [%]", fontsize=fontsize)
            ax[1].set_ylabel(r"Purity [%]", fontsize=fontsize)
            ax[2].set_ylabel(r"Failure rate [%]", fontsize=fontsize)

        # Plot the metrics
        metrics = ["C", "P", "F"]
        for i, metric in enumerate(metrics):
            self.plot_metric(
                metric,
                ax[i],
                show_legend=show_legend if i == 0 else False,
                linestyle=linestyle,
                color=color,
                alpha=alpha,
                interp_points=interp_points,
            )

        if show_legend:
            ax[0].legend(loc=loclegend, fontsize=legend_fontsize)
        if output_file_path is not None:
            plt.savefig(
                output_file_path, bbox_inches="tight", transparent=True, dpi=dpi
            )
        plt.show()

    def plot_color_distribution(
        self,
        ax,
        color1,
        color2,
        calibration_index,
        zl_index,
        show_ax_labels=True,
        xlim=(-1.5, 4.5),
        ylim=(-1.5, 4.5),
        plot_selection_criteria=True,
        plot_all_objects=None,
        plot_all_selected_objects="contour",
        plot_selected_objects="scatter",
        color_selection_criteria="b",
        color_all_objects="k",
        color_all_selected_objects="red",
        color_selected_objects="g",
        style_selection_criteria="--",
        style_all_objects="-",
        style_all_selected_objects="-",
        style_selected_objects="-",
        label_selection_criteria="Selection criteria",
        label_all_objects="All galaxies",
        label_all_selected_objects="All selected galaxies",
        label_selected_objects="Galaxies selected\nin this space",
        levels_2D=[0.68, 0.95],
        loclegend="lower right",
        legend_fontsize=13,
        labelsize=13,
        fontsize=15,
        zl_precision_in_legend="3",
        show_axes_grid=False,
        output_file_path=None,
        dpi=300,
    ):
        """
        Plot color-color distribution with selection criteria and object selection.

        This function plots a color-color space containing:
        - The distribution of all objects.
        - Selection criteria (linear cuts or grid regions) at a specific redshift threshold.
        - Selected objects based on all criteria (from all selection criteria in all color-color spaces).
        - Objects selected specifically in the current color-color space.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis object to plot on.
        color1, color2 : str
            Color names for x and y axes (e.g., 'g-r', 'g-i').
        calibration_index : int
            Index of the calibration object to use from self.calibrations.
        zl_index : int
            Index of the redshift threshold in DataModel.z_lim.
        show_ax_labels : bool, optional
            If true, show the ax labels.
        xlim, ylim : tuple of float, optional
            Axis limits for the plot.
        plot_selection_criteria : bool, optional
            Whether to plot the selection criteria lines.
        plot_all_objects : {None, "contour", "scatter"}, optional
            How to plot all objects from the entire calibration sample:
            None (don't plot), "contour" (density contours), or "scatter" (individual points).
        plot_all_selected_objects : {None, "contour", "scatter"}, optional
            How to plot all selected objects
            (from all selection criteria in all color-color spaces).
        plot_selected_objects : {None, "contour", "scatter"}, optional
            How to plot objects selected in the current color-color space.
        color_selection_criteria : str, optional
            Color for selection criteria lines/regions.
        color_all_objects : str, optional
            Color for all objects distribution.
        color_all_selected_objects : str, optional
            Color for all selected objects distribution.
        color_selected_objects : str, optional
            Color for objects selected in current space.
        style_selection_criteria : str, optional
            Either line style or marker style for selection criteria.
        style_all_objects : str, optional
            Either line style or marker style for all objects in the calibration sample.
        style_all_selected_objects : str, optional
            Either line style or marker style for all selected objects.
        style_selected_objects : str, optional
            Either line style or marker style for selected objects in current space.
        label_selection_criteria : str, optional
            Legend label for selection criteria.
        label_all_objects : str, optional
            Legend label for all objects.
        label_all_selected_objects : str, optional
            Legend label for all selected objects.
        label_selected_objects : str, optional
            Legend label for objects selected in current space.
        levels_2D : list of float, optional
            Probability levels for density contours (e.g., [0.68, 0.95]).
        loclegend : str, optional
            Location of the legend.
        legend_fontsize : int, optional
            Font size for legend text.
        labelsize : float, optional
            Size of the tick labels.
        fontsize : int, optional
            Font size for axis labels.
        zl_precision_in_legend : str, optional
            Number of decimal places for redshift value in legend title.
        show_axes_grid : bool, optional
            If True, show the axes tick grid. Default is False.
        output_file_path : str, optional
            Concatenation of the output file path and name.
            If None, the plot is not saved to file. Default is None.
        dpi : int, optional
            dpi parameter of `plt.savefig`.

        Notes
        -----
        The function computes color values from the calibration dataset and
        applies selection criteria from the calibration dictionary at the
        specified redshift threshold.

        The legend includes the redshift threshold value with specified precision.
        """
        check_plot_type(plot_all_objects, "plot_all_objects")
        check_plot_type(plot_all_selected_objects, "plot_all_selected_objects")
        check_plot_type(plot_selected_objects, "plot_selected_objects")

        ax.tick_params(direction="in", which="both", labelsize=labelsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if show_ax_labels:
            ax.set_xlabel(f"$({color1})$", fontsize=fontsize)
            ax.set_ylabel(f"$({color2})$", fontsize=fontsize)
        if show_axes_grid:
            ax.grid()

        calib = self.calibrations[calibration_index]

        z_l = calib.td.z_lim[zl_index]

        c1 = calib.dataset[color1[0]] - calib.dataset[color1[2]]
        c2 = calib.dataset[color2[0]] - calib.dataset[color2[2]]

        all_colors = [color[0] + "-" + color[1] for color in calib.colors]
        other_colors = [c for c in all_colors if c not in [color1, color2]]

        x_min, x_max = ax.get_xlim()

        if plot_selection_criteria or (plot_selected_objects is not None):

            label1_is_plotted = False
            label2_is_plotted = False

            mask_selection = np.full(len(c1), False)

            for key, entry in calib.dict.items():

                if key == "metadata":  # Skip metadata
                    continue

                contains_color1 = color1 in key
                contains_color2 = color2 in key
                contains_other = any(other_color in key for other_color in other_colors)

                # Only color1 OR color2, no other colors
                if (
                    contains_color1 ^ contains_color2
                ) and not contains_other:  # XOR operator

                    if z_l in entry["knots"]:

                        idx = np.where(np.unique(entry["knots"]) == z_l)[0][0]
                        c = entry["c_optimal"][idx]

                        # Mask
                        mask_selection |= calib.get_mask(zl_index, key)

                        # Selection criteria
                        if plot_selection_criteria:
                            if color1 in key:
                                ax.axvline(
                                    x=c,
                                    color=color_selection_criteria,
                                    linestyle=style_selection_criteria,
                                    label=(
                                        label_selection_criteria
                                        if not label1_is_plotted
                                        else None
                                    ),
                                )
                                label1_is_plotted = True
                            else:
                                ax.axhline(
                                    y=c,
                                    color=color_selection_criteria,
                                    linestyle=style_selection_criteria,
                                    label=(
                                        label_selection_criteria
                                        if not label1_is_plotted
                                        else None
                                    ),
                                )
                                label1_is_plotted = True

                # Both color1 AND color2
                if contains_color1 and contains_color2:

                    if z_l in entry["knots"]:

                        idx = np.where(np.unique(entry["knots"]) == z_l)[0][0]
                        c = entry["c_optimal"][idx]
                        s = entry["s_optimal"][idx]

                        # Mask
                        mask_selection |= calib.get_mask(zl_index, key)

                        # Selection criteria
                        if plot_selection_criteria:

                            x_line = np.linspace(x_min, x_max, 200)
                            if ">" in key:
                                if "-s*" in key:  # color1 > -s*(color2) + c
                                    y_line = (c - x_line) / s
                                else:  # color1 > s*(color2) + c
                                    y_line = (x_line - c) / s

                            else:  # '<' in key
                                if "-s*" in key:  # color1 < -s*(color2) + c
                                    y_line = (c - x_line) / s
                                else:  # color1 < s*(color2) + c
                                    y_line = (x_line - c) / s

                            ax.plot(
                                x_line,
                                y_line,
                                color=color_selection_criteria,
                                linestyle=style_selection_criteria,
                                label=(
                                    label_selection_criteria
                                    if not label1_is_plotted
                                    else None
                                ),
                            )
                            label1_is_plotted = True

            if plot_selected_objects is not None:
                c1_sel = c1[mask_selection]
                c2_sel = c2[mask_selection]
                if plot_selected_objects == "contour":
                    plot_distribution_contours(
                        ax,
                        c1_sel,
                        c2_sel,
                        50,
                        levels_2D,
                        color_selected_objects,
                        label_selected_objects,
                        style_selected_objects,
                    )
                else:
                    plot_distribution_scatter(
                        ax,
                        c1_sel,
                        c2_sel,
                        color_selected_objects,
                        label_selected_objects,
                    )

        if plot_all_selected_objects is not None:
            mask_selection = calib.get_mask(zl_index, "all")
            c1_sel = c1[mask_selection]
            c2_sel = c2[mask_selection]

            if plot_all_selected_objects == "contour":
                plot_distribution_contours(
                    ax,
                    c1_sel,
                    c2_sel,
                    50,
                    levels_2D,
                    color_all_selected_objects,
                    label_all_selected_objects,
                    style_all_selected_objects,
                )
            else:
                plot_distribution_scatter(
                    ax,
                    c1_sel,
                    c2_sel,
                    color_all_selected_objects,
                    label_all_selected_objects,
                )

        if plot_all_objects == "contour":
            plot_distribution_contours(
                ax,
                c1,
                c2,
                50,
                levels_2D,
                color_all_objects,
                label_all_objects,
                style_all_objects,
            )
        elif plot_all_objects == "scatter":
            plot_distribution_scatter(ax, c1, c2, color_all_objects, label_all_objects)

        handles, _ = ax.get_legend_handles_labels()
        if handles:  # Only create legend if there are handles
            ax.legend(
                loc=loclegend,
                title=r"$z_{\rm l} " + f"= {z_l:.{zl_precision_in_legend}f}$",
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
            )

        if output_file_path is not None:
            plt.savefig(
                output_file_path, bbox_inches="tight", transparent=True, dpi=dpi
            )

    def plot_all_color_distributions(
        self,
        calibration_index,
        zl_index,
        n_columns=3,
        figsize=(15, 20),
        wspace=0.35,
        hspace=0.35,
        xlim=(-1.5, 4.5),
        ylim=(-1.5, 4.5),
        plot_selection_criteria=True,
        plot_all_objects=None,
        plot_all_selected_objects="contour",
        plot_selected_objects="scatter",
        color_selection_criteria="b",
        color_all_objects="k",
        color_all_selected_objects="red",
        color_selected_objects="g",
        style_selection_criteria="--",
        style_all_objects="-",
        style_all_selected_objects="-",
        style_selected_objects="-",
        label_selection_criteria="Selection criteria",
        label_all_objects="All galaxies",
        label_all_selected_objects="All selected galaxies",
        label_selected_objects="Galaxies selected\nin this space",
        levels_2D=[0.68, 0.95],
        loclegend="lower right",
        legend_fontsize=10,
        labelsize=13,
        fontsize=15,
        zl_precision_in_legend="3",
        show_axes_grid=False,
        output_file_path=None,
        dpi=300,
    ):
        """
        Plot color-color distribution with selection criteria and object selection.

        This function creates a color-color plot showing:
        - The distribution of all objects.
        - Selection criteria (linear cuts or grid regions) at a specific redshift threshold.
        - Selected objects based on all criteria (from all selection criteria in all color-color spaces).
        - Objects selected specifically in the current color-color space.

        Parameters
        ----------
        calibration_index : int
            Index of the calibration object to use from self.calibrations.
        zl_index : int
            Index of the redshift threshold in DataModel.z_lim.
        n_columns : int, optional
            Number of subplot columns. Default is 3.
        figsize : tuple, optional
            Figure size. Default is (15, 4.5).
        wspace : float, optional
            Width space between the subplots. Default is 0.35.
        hspace : float, optional
            Height space between the subplots. Default is 0.35.
        xlim, ylim : tuple of float, or list thereof, optional
            Axis limits for the plot. If list, each element
            refers to the elements of the flattened axis array.
        plot_selection_criteria : bool, optional
            Whether to plot the selection criteria lines.
        plot_all_objects : {None, "contour", "scatter"}, optional
            How to plot all objects from the entire calibration sample:
            None (don't plot), "contour" (density contours), or "scatter" (individual points).
        plot_all_selected_objects : {None, "contour", "scatter"}, optional
            How to plot all selected objects
            (from all selection criteria in all color-color spaces).
        plot_selected_objects : {None, "contour", "scatter"}, optional
            How to plot objects selected in the current color-color space.
        color_selection_criteria : str, optional
            Color for selection criteria lines/regions.
        color_all_objects : str, optional
            Color for all objects distribution.
        color_all_selected_objects : str, optional
            Color for all selected objects distribution.
        color_selected_objects : str, optional
            Color for objects selected in current space.
        style_selection_criteria : str, optional
            Either line style or marker style for selection criteria.
        style_all_objects : str, optional
            Either line style or marker style for all objects in the calibration sample.
        style_all_selected_objects : str, optional
            Either line style or marker style for all selected objects.
        style_selected_objects : str, optional
            Either line style or marker style for selected objects in current space.
        label_selection_criteria : str, optional
            Legend label for selection criteria.
        label_all_objects : str, optional
            Legend label for all objects.
        label_all_selected_objects : str, optional
            Legend label for all selected objects.
        label_selected_objects : str, optional
            Legend label for objects selected in current space.
        levels_2D : list of float, optional
            Probability levels for density contours (e.g., [0.68, 0.95]).
        loclegend : str, optional
            Location of the legend.
        legend_fontsize : int, optional
            Font size for legend text.
        labelsize : float, optional
            Size of the tick labels.
        fontsize : int, optional
            Font size for axis labels.
        zl_precision_in_legend : str, optional
            Number of decimal places for redshift value in legend title.
        show_axes_grid : bool, optional
            If True, show the axes tick grid. Default is False.
        output_file_path : str, optional
            Concatenation of the output file path and name.
            If None, the plot is not saved to file. Default is None.
        dpi : int, optional
            dpi parameter of `plt.savefig`.
        """
        calib = self.calibrations[calibration_index]

        color_spaces = list(combinations(calib.colors, 2))

        n_color_spaces = len(color_spaces)
        n_rows = int(np.ceil(n_color_spaces / n_columns))
        fig, ax = plt.subplots(
            n_rows,
            n_columns,
            figsize=figsize,
            gridspec_kw={"wspace": wspace, "hspace": hspace},
        )

        ax_flat = np.array(ax).flatten()
        for i in range(n_color_spaces, len(ax_flat)):
            fig.delaxes(ax_flat[i])  # delete unused axes

        xlim = check_ax_limits(xlim, "xlim", n_color_spaces)
        ylim = check_ax_limits(ylim, "ylim", n_color_spaces)

        for i in range(n_color_spaces):

            color1 = color_spaces[i][0][0] + "-" + color_spaces[i][0][1]
            color2 = color_spaces[i][1][0] + "-" + color_spaces[i][1][1]

            self.plot_color_distribution(
                ax_flat[i],
                color1,
                color2,
                calibration_index,
                zl_index,
                xlim=xlim[i],
                ylim=ylim[i],
                plot_selection_criteria=plot_selection_criteria,
                plot_all_objects=plot_all_objects,
                plot_all_selected_objects=plot_all_selected_objects,
                plot_selected_objects=plot_selected_objects,
                color_selection_criteria=color_selection_criteria,
                color_all_objects=color_all_objects,
                color_all_selected_objects=color_all_selected_objects,
                color_selected_objects=color_selected_objects,
                style_selection_criteria=style_selection_criteria,
                style_all_objects=style_all_objects,
                style_all_selected_objects=style_all_selected_objects,
                style_selected_objects=style_selected_objects,
                label_selection_criteria=label_selection_criteria if i == 0 else None,
                label_all_objects=label_all_objects if i == 0 else None,
                label_all_selected_objects=(
                    label_all_selected_objects if i == 0 else None
                ),
                label_selected_objects=label_selected_objects if i == 0 else None,
                levels_2D=levels_2D,
                loclegend=loclegend,
                legend_fontsize=legend_fontsize,
                labelsize=labelsize,
                fontsize=fontsize,
                zl_precision_in_legend=zl_precision_in_legend,
                show_axes_grid=show_axes_grid,
                output_file_path=None,
            )

        if output_file_path is not None:
            plt.savefig(
                output_file_path, bbox_inches="tight", transparent=True, dpi=dpi
            )
        plt.show()

    def plot_linear_cut_parameters(
        self,
        calibration_index,
        cut_key,
        interp_points=None,
        figsize=(14, 4.5),
        wspace=0.35,
        color_measure="k",
        color_interp="r",
        style_measure="o",
        style_interp="--",
        label_measure="Optimal values",
        label_interp="Interpolation",
        loclegend="upper right",
        legend_fontsize=13,
        labelsize=13,
        fontsize=17,
        output_file_path=None,
        dpi=300,
    ):
        """
        Plot color condition parameters as a function of limiting redshift.

        One subplot is produced in the slope parameter is not present
        in the chosen color condition. Two subplots are produced otherwise.

        NOTE: the sign of the slope parameter is always positive in the plot.
        The minus sign, if present, is reported in the legend.

        Parameters
        ----------
        calibration_index : int
            Index of the calibration object to use from self.calibrations.
        cut_key : str
            Key corresponding to the optimal color condition.
            The list of optimal color conditions can be obtained
            using `CalibrateLinear.get_keys()`.
        interp_points : array-like, optional
            Limiting redshift points. If not None, derive the parameters
            at these points using interpolations. Default is None.
        figsize : tuple, optional
            Figure size. Default is (14, 4.5).
        wspace : float, optional
            Width space between the subplots. Default is 0.35.
        color_measure : str, optional
            Color of the optimal parameter measurements.
        color_interp : str, optional
            Color of the interpolated line (if `interp_points` is not None).
        style_measure : str, optional
            Marker style of the measured optimal values.
        style_interp : str, optional
            Line style of the interpolated values.
        label_measure : str, optional
            Legend label of the measurement points.
        loclegend : str, optional
            Location of the legend.
        legend_fontsize : int, optional
            Font size for legend text.
        labelsize : float, optional
            Size of the tick labels.
        fontsize : int, optional
            Base font size for axis labels.
        output_file_path : str, optional
            Concatenation of the output file path and name.
            If None, the plot is not saved to file. Default is None.
        dpi : int, optional
            dpi parameter of `plt.savefig`.
        """
        calib = self.calibrations[calibration_index]
        assert isinstance(
            calib, CalibrateLinear
        ), f"Expected CalibrateLinear, got {type(calib).__name__}"

        entry = calib.dict[cut_key]
        z_lim = np.unique(entry["knots"])

        if "s" in cut_key:
            fig, ax = plt.subplots(
                1, 2, figsize=figsize, gridspec_kw={"wspace": wspace}
            )
            ax[1].set_ylabel(r"$s$", fontsize=fontsize)
            ax[1].scatter(
                z_lim,
                entry["s_optimal"],
                c=color_measure,
                marker=style_measure,
            )
            if interp_points is not None:
                s_interp = BSpline(
                    entry["knots"],
                    entry["s_coefficients"],
                    entry["order"],
                    extrapolate=False,
                )
                ax[1].plot(
                    interp_points,
                    s_interp(interp_points),
                    color=color_interp,
                    linestyle=style_interp,
                )
        else:
            fig, ax = plt.subplots(figsize=figsize, gridspec_kw={"wspace": wspace})

        ax_flat = np.array(ax).flatten()
        for ax in ax_flat:
            ax.tick_params(direction="in", which="both", labelsize=labelsize)
            ax.set_xlabel(r"$z_{\rm l}$", fontsize=fontsize)
        ax_flat[0].set_ylabel(r"$c$", fontsize=fontsize)

        ax_flat[0].scatter(
            z_lim,
            entry["c_optimal"],
            c=color_measure,
            marker=style_measure,
            label=label_measure,
        )
        if interp_points is not None:
            c_interp = BSpline(
                entry["knots"],
                entry["c_coefficients"],
                entry["order"],
                extrapolate=False,
            )
            ax_flat[0].plot(
                interp_points,
                c_interp(interp_points),
                color=color_interp,
                linestyle=style_interp,
                label=label_interp,
            )

        ax_flat[0].legend(
            loc=loclegend,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            title=r"$" + cut_key.replace("*", "\,") + "$",
        )

        if output_file_path is not None:
            plt.savefig(
                output_file_path, bbox_inches="tight", transparent=True, dpi=dpi
            )
        plt.show()
