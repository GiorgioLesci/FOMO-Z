"""
Helper functions for data visualization.

This module provides auxiliary functions for data visualization.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

from .statistics import distribution_contours


def normalize_plot_style(value, name, n_calibrations):
    """
    Normalize plot style parameters to a list of appropriate length.

    Parameters
    ----------
    value : Any
        The style value to normalize (color, linestyle, etc.).
    name : str
        Name of the parameter for error messages.
    n_calibrations : list
        List of calibration objects.

    Returns
    -------
    List
        List of values with length matching number of calibrations.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != n_calibrations:
            raise ValueError(
                f"Number of {name}s ({len(value)}) must match "
                f"the number of calibrations ({n_calibrations})."
            )
        return value
    elif isinstance(value, str):
        return [value]
    else:
        raise TypeError("{name} must be either string, list, or tuple.")


def plot_distribution_contours(ax, x, y, nbins, levels_2D, color, label, linestyle):
    """
    Plot 2D density contours on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to plot on.
    x, y : array-like
        Data coordinates for contour computation.
    nbins : int
        Number of bins for the 2D histogram.
    levels_2D : list of float
        Probability levels for contours (e.g., [0.68, 0.95]).
    color : str
        Color for the contour lines.
    label : str
        Label for the legend entry.
    linestyle : str or list of str
        Line style(s) for the contours.

    Notes
    -----
    Adds a dummy vertical line at x=1e16 for legend purposes only.
    The actual data is displayed as density contours.
    """
    X, Y, H_norm, level_density = distribution_contours(x, y, nbins, levels_2D)
    ax.contour(X, Y, H_norm.T, levels=level_density, colors=color, linestyles=linestyle)
    ax.axvline(1.0e16, linewidth=2, color=color, label=label)  # for legend only


def plot_distribution_scatter(ax, x, y, color, label):
    """
    Plot data points as a scatter plot on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to plot on.
    x, y : array-like
        Data coordinates to scatter.
    color : str
        Color for the scatter points.
    label : str
        Label for the legend entry.

    Notes
    -----
    Adds a dummy point at (1e16, 1e16) for legend purposes only.
    This point is placed far outside typical data ranges to be invisible.
    """
    ax.scatter(x, y, c=color, s=2)
    ax.scatter(1.0e16, 1.0e16, s=20, c=color, label=label)  # for legend only


def check_plot_type(plot_type, var_name):
    """
    Validate plot type parameter.

    Parameters
    ----------
    plot_type : any
        Plot type value to validate.
    var_name : str
        Name of the variable being checked (for error message).

    Raises
    ------
    ValueError
        If plot_type is not None, "contour", or "scatter".

    Notes
    -----
    This function checks that plot_type is one of the supported values
    and provides a descriptive error message if not.
    """
    if plot_type not in [None, "contour", "scatter"]:
        raise ValueError(
            f"The {var_name} input variable equals a not upported value ({plot_type})."
        )


def check_ax_limits(limits, name, expected_length):
    """
    Normalize axis limits to list format for multiple subplots.

    Parameters
    ----------
    limits : tuple or list
        Single tuple (min, max) for all subplots, or list of tuples.
    name : str
        Variable name for error messages.
    expected_length : int
        Number of subplots.

    Returns
    -------
    list
        List of limit tuples.
    """
    if isinstance(limits, tuple) and len(limits) == 2:
        return [limits] * expected_length
    elif isinstance(limits, list):
        if len(limits) != expected_length:
            raise ValueError(
                f"{name} list length ({len(limits)}) must match number of subplots ({expected_length})"
            )
        return limits
    else:
        raise TypeError("{name} must be a tuple (min, max) or list of tuples.")
