"""
AUXILIARY FUNCTIONS.

This module provides auxiliary functions for metrics computation,
data visualization, and data conversion.

:Author: Giorgio Francesco Lesci <giorgio.lesci2 [at] unibo [dot] it>
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(mask, redshift, z_l):
    """
    Compute classification metrics (TP, FP, FN, TN) based on a boolean mask
    and a redshift threshold.

    The function evaluates how well a given mask separates objects above and
    below a specified limiting redshift `z_l`. It compares the selected and
    unselected samples against the threshold to compute true/false positives
    and true/false negatives.

    Parameters
    ----------
    mask : array-like of bool
        Boolean mask indicating which objects are selected (True) and which
        are not (False).
    redshift : array-like of float
        Array of redshift values corresponding to the objects.
    z_l : float
        Limiting redshift used to classify objects
        as "background" (z > z_l) or "foreground" (z < z_l).

    Returns
    -------
    TP : int
        True Positives – number of selected objects with redshift > z_l.
    FP : int
        False Positives – number of selected objects with redshift < z_l.
    FN : int
        False Negatives – number of unselected objects with redshift > z_l.
    TN : int
        True Negatives – number of unselected objects with redshift < z_l.
    """
    redshift_sel = redshift[mask]
    redshift_unsel = redshift[~mask]

    TP = np.count_nonzero(redshift_sel > z_l)
    FP = np.count_nonzero(redshift_sel < z_l)
    FN = np.count_nonzero(redshift_unsel > z_l)
    TN = np.count_nonzero(redshift_unsel < z_l)

    return TP, FP, FN, TN

def compute_for_simple(args):
    """
    Compute classification metrics for a simple threshold on a single color.

    The function applies a threshold `c` on a color array,
    generating two masks:
    - `mask_up` for objects with color > c
    - `mask_low` for objects with color < c

    It then computes the metrics (TP, FP, FN, TN) for both masks relative to
    a given redshift threshold.

    Parameters
    ----------
    args : tuple
        A tuple containing:
        - c : float
            Threshold value applied to the color array.
        - color : array-like of float
            Array of color values for the objects.
        - redshift : array-like of float
            Array of redshift values corresponding to the objects.
        - z_l : float
            Limiting redshift used for classification.

    Returns
    -------
    list of int
        Concatenated list of metrics in the following order:
        [TP_up, FP_up, FN_up, TN_up, TP_low, FP_low, FN_low, TN_low]
        where the suffix `_up` refers to the mask `color > c` and `_low`
        refers to `color < c`.
    """
    c, color, redshift, z_l = args
    mask_up = color > c
    mask_low = color < c

    return [
        *compute_metrics(mask_up, redshift, z_l),
        *compute_metrics(mask_low, redshift, z_l),
    ]

def compute_for_cross(args):
    """
    Compute classification metrics for linear cuts in a 2D color-color space.

    This function applies four linear decision boundaries of the form
    `color1 > ±s * color2 + c` and `color1 < ±s * color2 + c`.
    Each cut creates a boolean mask dividing the color-color space, and the
    function evaluates classification performance (TP, FP, FN, TN) for each.

    Parameters
    ----------
    args : tuple
        A tuple containing:
        - s : float
            Slope of the linear cuts applied in color-color space.
        - c : float
            Intercept (constant offset) for the linear cuts.
        - color1 : array-like of float
            First color feature array.
        - color2 : array-like of float
            Second color feature array.
        - redshift : array-like of float
            Array of redshift values corresponding to the objects.
        - z_l : float
            Limiting redshift used for classification.

    Returns
    -------
    list of tuple
        A list of length 4, each entry being a tuple of classification metrics:
        (TP, FP, FN, TN).
        The order corresponds to the cuts:
        1. `color1 >  s * color2 + c`
        2. `color1 > -s * color2 + c`
        3. `color1 <  s * color2 + c`
        4. `color1 < -s * color2 + c`
    """
    s, c, color1, color2, redshift, z_l = args

    cuts = [
        color1 > s * color2 + c,
        color1 > -s * color2 + c,
        color1 < s * color2 + c,
        color1 < -s * color2 + c,
    ]

    return [compute_metrics(mask, redshift, z_l) for mask in cuts]

def set_plot():
    """
    Create and configure a figure with three side-by-side subplots for
    classification performance metrics.

    The function generates a 1x3 grid of subplots intended for plotting:
    - Completeness (left panel)
    - Purity (middle panel)
    - Foreground failure rate (right panel)

    Returns
    -------
    ax : ndarray of matplotlib.axes.Axes
        Array of 3 subplot axes: [ax[0], ax[1], ax[2]]
        corresponding to the panels for completeness, purity, and
        foreground failure rate, respectively.
    """
    fig, ax = plt.subplots(
        1, 3, figsize=(15, 4.5), sharex="row", gridspec_kw={"wspace": 0.3}
    )
    for i in range(len(ax)):
        ax[i].xaxis.set_tick_params(direction="in", which="both", labelsize=13)
        ax[i].yaxis.set_tick_params(direction="in", which="both", labelsize=13)
        # ax[i].grid()
        ax[i].set_xlabel(r"$z_{\rm l}$", fontsize=15)
    ax[0].set_ylabel(r"Completeness [%]", fontsize=15)
    ax[1].set_ylabel(r"Purity [%]", fontsize=15)
    ax[2].set_ylabel(r"Foreground failure rate [%]", fontsize=15)

    return ax

def make_yaml_serializable(obj):
    """
    Recursively convert complex or NumPy-based data types into
    plain Python types that can be safely serialized to YAML.

    This function traverses nested data structures (dictionaries,
    lists, tuples) and converts objects that are not natively
    supported by the YAML safe dumper (such as NumPy arrays or
    NumPy scalar types) into equivalent Python representations.

    Specifically:
        - ``numpy.ndarray`` is converted to ``list``
        - ``numpy.generic`` (e.g. ``numpy.float64``, ``numpy.int32``)
          is converted to corresponding Python ``float`` or ``int``
        - Nested containers (``dict``, ``list``, ``tuple``)
          are processed recursively
        - All other objects are returned unchanged

    Parameters
    ----------
    obj : any
        The input object (typically a dictionary or nested structure)
        to be converted into a YAML-safe form.

    Returns
    -------
    any
        A copy of the input object containing only built-in Python types
        (``dict``, ``list``, ``tuple``, ``str``, ``int``, ``float``,
        ``bool``, ``None``), suitable for serialization.
    """
    if isinstance(obj, dict):
        return {k: make_yaml_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_yaml_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_yaml_serializable(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # catches np.float32, np.int64, etc.
        return obj.item()
    else:
        return obj

def float_representer(dumper, value):
    """
    Custom YAML representer for floating-point numbers.

    This function defines how floating-point values are converted to text
    when dumping data to a YAML file using ``yaml.safe_dump``.

    Parameters
    ----------
    dumper : yaml.Dumper
        The YAML dumper instance provided by PyYAML during serialization.
    value : float
        The floating-point number to represent in YAML.

    Returns
    -------
    yaml.ScalarNode
        A YAML scalar node containing the formatted float as a string.

    Notes
    -----
        Precision hardcoded as ``{value:.6g}``
    """
    if value == 0.0:
        return dumper.represent_scalar("tag:yaml.org,2002:int", "0")

    elif value.is_integer():
        text = f"{int(value)}"
        return dumper.represent_scalar("tag:yaml.org,2002:int", text)

    else:
        text = f"{value:.6g}"
        return dumper.represent_scalar("tag:yaml.org,2002:float", text)
