"""
Data visualization module.

This module provides functions for data visualization.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

from typing import Union, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from .tools import compute_metrics
from .calibrate_linear import CalibrateLinear
from .select_object import SelectObject


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

    def _normalize_plot_style(self, value, name):
        """
        Normalize plot style parameters to a list of appropriate length.

        Parameters
        ----------
        value : Any
            The style value to normalize (color, linestyle, etc.).
        name : str
            Name of the parameter for error messages.

        Returns
        -------
        List
            List of values with length matching number of calibrations.
        """
        if isinstance(value, (list, tuple)):
            if len(value) != self.n_calibrations:
                raise ValueError(
                    f"Number of {name}s ({len(value)}) must match "
                    f"the number of calibrations ({self.n_calibrations})."
                )
            return value
        elif isinstance(value, str):
            return [value]
        else:
            raise TypeError("{name} must be either string, list, or tuple.")

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

        linestyle = self._normalize_plot_style(linestyle, "linestyle")
        color = self._normalize_plot_style(color, "color")

        for i, calibration in enumerate(self.calibrations):
            z_lim = calibration.dm.z_lim
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

                dataset = calibration.dm.dataset
                sel.process(interp_points, dataset)

                y = []
                for j, z_l in enumerate(interp_points):
                    target_mask = calibration.dm.get_target_mask(dataset, [z_l])
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
        show_grid=False,
        alpha=0.3,
        interp_points=None,
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
        show_grid : bool, optional
            If True, show the axis tick grid. Default is False.
        alpha : float, optional
            alpha parameter of the 68% confidence interval. Default is 0.3.
        interp_points : array-like, optional
            Limiting redshift points. If not None, derive the metrics at these points
            using interpolations. Valid only for CalibrateLinear instances. Default is None.
        """
        # Set up the plot
        fig, ax = plt.subplots(
            1, 3, figsize=figsize, sharex="row", gridspec_kw={"wspace": wspace}
        )
        for i in range(len(ax)):
            ax[i].xaxis.set_tick_params(
                direction="in", which="both", labelsize=labelsize
            )
            ax[i].yaxis.set_tick_params(
                direction="in", which="both", labelsize=labelsize
            )
            if show_grid:
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
        plt.show()
