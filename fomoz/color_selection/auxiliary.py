"""
AUXILIARY FUNCTIONS.

This module provides auxiliary functions for metrics computation,
interpolations, data visualization, and data conversion.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


def get_target_mask(dataset, z_lim, is_zlim_lower, additional_rules):
    """
    Create a mask identifying galaxies that satisfy a set of conditions.

    This function generates a 2D boolean mask array where each row corresponds
    to a different redshift limit and each column corresponds to a galaxy in the
    dataset. A value of `True` indicates that a galaxy meets all selection criteria
    for that particular redshift limit.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Input dataset containing galaxy properties. Must include a 'redshift' column
        and any additional properties referenced in `additional_rules`.
    z_lim : array-like of float
        Array of redshift thresholds to apply. Each value defines a separate
        selection criterion.
    is_zlim_lower : bool
        If True, `z_lim` values are treated as lower limits. If False, `z_lim`
        values are treated as upper limits.
    additional_rules : list of lists or None
        Additional selection rules beyond redshift. Each rule is a list with format:
        (property_name, operator, threshold). All rules are combined with AND logic.
        Supported operators: ">", "<", ">=", "<=", "==".

        Example:
        [("logM", ">", 10.0), ("u", "<", 24.0)] selects galaxies with
        stellar mass > 10^10 solar masses AND u-band magnitude < 24.0.

    Returns
    -------
    target_mask : numpy.ndarray of bool
        2D boolean array with shape (len(z_lim), len(dataset)).
        target_mask[i, j] = True indicates that galaxy j satisfies all selection
        criteria for redshift limit z_lim[i].

    Notes
    -----
    - The function applies all selection criteria with AND logic: a galaxy must
      satisfy the redshift condition AND all additional rules to be marked as True.
    - The returned mask can be used directly for calculating completeness, purity,
      and other selection metrics.
    """
    target_mask = np.full((len(z_lim), len(dataset)), False, dtype=bool)
    for i, zz in enumerate(z_lim):

        # Limiting redshift
        if is_zlim_lower:
            target_mask[i] = dataset["redshift"] > zz
        else:
            target_mask[i] = dataset["redshift"] < zz

        # Additional rules
        if additional_rules is None:
            continue
        for property_name, operator, threshold in additional_rules:

            data = dataset[property_name]

            if operator == ">":
                target_mask[i] &= data > threshold
            elif operator == "<":
                target_mask[i] &= data < threshold
            elif operator == ">=":
                target_mask[i] &= data >= threshold
            elif operator == "<=":
                target_mask[i] &= data <= threshold
            elif operator == "==":
                target_mask[i] &= data == threshold

    return target_mask


def compute_metrics(mask, target_mask):
    """
    Compute classification metrics (TP, FP, FN, TN).

    The function computes true/false positives and true/false negatives
    based on the input masks.

    Parameters
    ----------
    mask : array-like of bool
        Boolean mask of selected objects by color cuts.
    target_mask : np.ndarray
        Boolean mask of target objects (satisfying redshift + additional rules).

    Returns
    -------
    TP : int
        True Positives.
    FP : int
        False Positives.
    FN : int
        False Negatives.
    TN : int
        True Negatives.
    """
    if mask.shape != target_mask.shape:
        raise ValueError(
            f"Mask shapes do not match: mask has shape {mask.shape}, "
            f"target_mask has shape {target_mask.shape}"
        )

    TP = np.count_nonzero(mask & target_mask)
    FP = np.count_nonzero(mask & ~target_mask)
    FN = np.count_nonzero(~mask & target_mask)
    TN = np.count_nonzero(~mask & ~target_mask)

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
        - target_mask : np.ndarray
            Boolean mask of target objects (satisfying redshift + additional rules).

    Returns
    -------
    list of int
        Concatenated list of metrics in the following order:
        [TP_up, FP_up, FN_up, TN_up, TP_low, FP_low, FN_low, TN_low]
        where the suffix `_up` refers to the mask `color > c` and `_low`
        refers to `color < c`.
    """
    c, color, target_mask = args
    mask_up = color > c
    mask_low = color < c

    return [
        *compute_metrics(mask_up, target_mask),
        *compute_metrics(mask_low, target_mask),
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
        - target_mask : np.ndarray
            Boolean mask of target objects (satisfying redshift + additional rules).

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
    s, c, color1, color2, target_mask = args

    cuts = [
        color1 > s * color2 + c,
        color1 > -s * color2 + c,
        color1 < s * color2 + c,
        color1 < -s * color2 + c,
    ]

    return [compute_metrics(mask, target_mask) for mask in cuts]


def c_domain_given_s(color1, color2, s, q_low=0.01, q_high=0.99):
    """
    Get the domain of `c`.

    Estimate the domain for the intercept parameter `c` in
    color-color conditions of the form:

        color1 >  +s * color2 + c
        color1 <  +s * color2 + c
        color1 >  -s * color2 + c
        color1 <  -s * color2 + c

    The input slope values `s` are always positive.

    The inequality can be rewritten as:
        c <  color1 - s' * color2,
    Consequently, all meaningful values of `c` lie
    within the empirical range of the transformed variables:
        X(+s) = color1 - s * color2
        X(-s) = color1 + s * color2

    This function combines both transformations and computes lower and
    upper quantile bounds to exclude extreme outliers while retaining the
    full range of relevant `c` values.

    Parameters
    ----------
    color1, color2 : ndarray (1D)
        Input color arrays (may contain NaNs). Must have matching shape.
    s : float
        Positive slope magnitude used to evaluate both +s and -s conditions.
    q_low, q_high : float
        Lower and upper quantiles for robust domain estimation
        (0 < q_low < q_high < 1). Default: [0.01, 0.99].

    Returns
    -------
    c_min, c_max : float
        Lower and upper bounds for the valid c domain.
    """
    color1 = np.asarray(color1, dtype=np.float64)
    color2 = np.asarray(color2, dtype=np.float64)

    if color1.shape != color2.shape:
        raise ValueError("color1 and color2 must have the same shape")

    # Values relevant for +s and -s conditions
    vals = np.concatenate(
        [
            color1 - s * color2,  # +s
            color1 + s * color2,  # -s
        ]
    )

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -5.0, 5.0

    lo = float(np.quantile(vals, q_low))
    hi = float(np.quantile(vals, q_high))

    return lo, hi


def build_nested_dict(cuts):
    """
    Build a nested dictionary from a list of color cuts.

    This utility organizes a flat list of cuts into a two-level nested
    dictionary structure based on their associated color pairs. Each
    cut is grouped first by its `base_pair`, then by its `comp_pair`.

    Parameters
    ----------
    cuts : dict
        Dictionary of cut definitions, where keys are tuples of the form
        (base_pair, comp_pair, condition_index) and values are dictionaries
        containing cut parameters. Each value dictionary must contain:
        - "c" : float
            Intercept parameter of the cut
        - "completeness" : float  
            Completeness value of the cut
        - "s" : float, optional
            Slope parameter (present only for cross-color cuts)

    Returns
    -------
    dict
        A nested dictionary of the form:
        {
            base_pair1: {
                comp_pair1: [cut1, cut2, ...],
                comp_pair2: [...],
                ...
            },
            base_pair2: {
                ...
            },
            ...
        }
    """
    nested = {}
    for key, cut in cuts.items():
        base_pair, comp_pair, cond_index = key
        nested.setdefault(base_pair, {}).setdefault(comp_pair, []).append({
            "condition_index": cond_index,
            "c": cut["c"],
            "s": cut.get("s"),  # None for simple cuts
            "completeness": cut["completeness"]
        })
    return nested


def restructure_dict(input_dict, bands, colors, z_lim, purity_min):
    """
    Restructure the nested dictionary of color conditions.

    This method converts the input dictionary of cuts (organized by
    redshift bins, base pairs, and comparison pairs) into a flatter
    representation keyed by human-readable string expressions of the
    color conditions.

    Parameters
    ----------
    input_dict : dict
        Nested dictionary of the form::
            {
                z_l: {
                    color1: {
                        color2: [list of cut dictionaries]
                    }
                }
            }

        Each cut dictionary must include:
        - "condition_index" : int
            Index specifying which inequality is applied (0–3).
        - "c" : float
            Intercept of the cut.
        - "s" : float, optional
            Slope of the cut (present only for cross-color cuts).

    Returns
    -------
    dict
        Restructured dictionary keyed by string expressions of the cuts,
        e.g. ``"g-r > s*(r-i) + c"``. Each entry stores:
        - "z_l" : list of float
            Redshift thresholds associated with the cut.
        - "c" : list of float
            Intercepts of the cut across redshifts.
        - "s" : list of float, optional
            Slopes of the cut (if applicable).
    """
    dict_restructured = {}
    dict_restructured["metadata"] = {}
    dict_restructured["metadata"]["bands"] = bands
    dict_restructured["metadata"]["z_l"] = [float(z) for z in z_lim]
    dict_restructured["metadata"]["purity_min"] = [float(P) for P in purity_min]

    for i in range(len(colors)):
        for j in range(i, len(colors)):
            mag1, mag2 = colors[i][0], colors[i][1]
            mag3, mag4 = colors[j][0], colors[j][1]
            color1 = (mag1, mag2)
            color2 = (mag3, mag4)
            for z_l in z_lim:

                if color1 in input_dict[z_l]:

                    if color2 in input_dict[z_l][color1]:

                        dic = input_dict[z_l][color1][color2]
                        for k in range(len(dic)):

                            if color1 == color2 and dic[k]["condition_index"] == 0:
                                expr = f"{mag1}-{mag2}>c"
                            elif color1 == color2 and dic[k]["condition_index"] == 1:
                                expr = f"{mag1}-{mag2}<c"
                            elif color1 != color2 and dic[k]["condition_index"] == 0:
                                expr = f"{mag1}-{mag2}>s*({mag3}-{mag4})+c"
                            elif color1 != color2 and dic[k]["condition_index"] == 1:
                                expr = f"{mag1}-{mag2}>-s*({mag3}-{mag4})+c"
                            elif color1 != color2 and dic[k]["condition_index"] == 2:
                                expr = f"{mag1}-{mag2}<s*({mag3}-{mag4})+c"
                            elif color1 != color2 and dic[k]["condition_index"] == 3:
                                expr = f"{mag1}-{mag2}<-s*({mag3}-{mag4})+c"

                            if expr not in dict_restructured:
                                dict_restructured[expr] = {}

                            if "z_l" not in dict_restructured[expr]:
                                dict_restructured[expr]["z_l"] = []
                            dict_restructured[expr]["z_l"].append(float(z_l))

                            if "c" not in dict_restructured[expr]:
                                dict_restructured[expr]["c"] = []
                            dict_restructured[expr]["c"].append(dic[k]["c"])

                            if "s" in expr:
                                if "s" not in dict_restructured[expr]:
                                    dict_restructured[expr]["s"] = []
                                dict_restructured[expr]["s"].append(dic[k]["s"])
    input_dict.clear()
    input_dict.update(dict_restructured)


def interpolate_parameters(input_dict, k=1):
    """
    Interpolate the color cut parameters across redshift bins using splines.

    For each color cut expression in the input dictionary, fits a
    `scipy.interpolate.UnivariateSpline` to the intercepts (``c``)
    and, if present, the slopes (``s``), as functions of the redshift
    thresholds (``z_l``). The fitted spline parameters (knots,
    coefficients, and order) are stored back into the dictionary.

    Parameters
    ----------
    input_dict : dict
        Dictionary of color cuts. Each entry must contain:
        - ``"z_l"`` : list of float
            Redshift thresholds.
        - ``"c"`` : list of float
            Intercepts at each redshift.
        - ``"s"`` : list of float, optional
            Slopes at each redshift (for cross-color cuts).
    k : int, optional
        Requested degree of the spline. Default is 1 (linear).
        The actual degree is adjusted per case:
        - If there are fewer than ``k+1`` points, the effective degree
          is reduced to ``n_points-1``.
        - If there is only one point, interpolation is skipped.

    Returns
    -------
    None
        Updates ``input_dict`` in place by renaming ``"c"``
        as ``"c_optimal"`` and ``"s"`` as ``"s_optimal"``,
        and adding:
        - ``"knots"``, ``"order"``, ``"c_coefficients"``
        - ``"s_coefficients"`` (if ``s`` is present)
    """
    for key in input_dict.keys():

        # skip metadata
        if key == "metadata":
            continue

        # rename c and s keys
        input_dict[key]["c_optimal"] = input_dict[key].pop("c")
        if "s" in input_dict[key]:
            input_dict[key]["s_optimal"] = input_dict[key].pop("s")

        # skip if there's only 1 point
        n_points = len(input_dict[key]["c_optimal"])
        if n_points <= 1:
            del input_dict[key]["z_l"]
            continue

        # adjust spline degree per case
        k_eff = min(k, n_points - 1)

        # Interpolate c
        c_spline = UnivariateSpline(
            input_dict[key]["z_l"],
            input_dict[key]["c_optimal"],
            s=0.0,
            k=k_eff,
            ext=2,
        )
        t, c, order = c_spline._eval_args
        input_dict[key]["knots"] = list(t)
        input_dict[key]["order"] = int(order)
        input_dict[key]["c_coefficients"] = list(c)

        # Interpolate s (if present)
        if "s_optimal" in input_dict[key]:
            s_spline = UnivariateSpline(
                input_dict[key]["z_l"],
                input_dict[key]["s_optimal"],
                s=0.0,
                k=k_eff,
                ext=2,
            )
            t, c, order = s_spline._eval_args
            input_dict[key]["s_coefficients"] = c

        # After all the interpolation is done, remove z_l
        del input_dict[key]["z_l"]


def set_plot():
    """
    Create a figure with three side-by-side subplots.

    The function generates a 1x3 grid of subplots intended for plotting:
    - Completeness (left panel)
    - Purity (middle panel)
    - Failure rate (right panel)

    Returns
    -------
    ax : ndarray of matplotlib.axes.Axes
        Array of 3 subplot axes: [ax[0], ax[1], ax[2]]
        corresponding to the panels for completeness, purity, and
        failure rate, respectively.
    """
    fig, ax = plt.subplots(
        1, 3, figsize=(15, 4.5), sharex="row", gridspec_kw={"wspace": 0.35}
    )
    for i in range(len(ax)):
        ax[i].xaxis.set_tick_params(direction="in", which="both", labelsize=13)
        ax[i].yaxis.set_tick_params(direction="in", which="both", labelsize=13)
        # ax[i].grid()
        ax[i].set_xlabel(r"$z_{\rm l}$", fontsize=15)
    ax[0].set_ylabel(r"Completeness [%]", fontsize=15)
    ax[1].set_ylabel(r"Purity [%]", fontsize=15)
    ax[2].set_ylabel(r"Failure rate [%]", fontsize=15)

    return ax


def make_yaml_serializable(obj):
    """
    Convert complex or NumPy-based data types into YAML serializables.

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
    Get custom YAML representer for floating-point numbers.

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
