"""
Compute masks, interpolations, true/false positives, and true/false negatives.

This module provides auxiliary functions for the computation of
masks, metrics, and interpolations.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

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
        Supported operators: ">", "<", ">=", "<=", "==", "!=".

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
            elif operator == "!=":
                target_mask[i] &= data != threshold

    return target_mask

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
        as ``"c_coefficients"`` and ``"s"`` as ``"s_coefficients"``,
        and adding ``"knots"`` and ``"order"``
    """
    for key in input_dict.keys():

        # skip metadata
        if key == "metadata":
            continue

        # rename c and s keys
        input_dict[key]["c_optimal"] = input_dict[key]["c"]
        input_dict[key]["c_coefficients"] = input_dict[key].pop("c")
        if "s" in input_dict[key]:
            input_dict[key]["s_optimal"] = input_dict[key]["s"]
            input_dict[key]["s_coefficients"] = input_dict[key].pop("s")

        # skip if there's only 1 point
        n_points = len(input_dict[key]["c_coefficients"])
        if n_points <= 1:
            input_dict[key]["knots"] = input_dict[key].pop("z_l")
            input_dict[key]["c_coefficients"] = input_dict[key]["c_coefficients"][0]
            if "s_coefficients" in input_dict[key]:
                input_dict[key]["s_coefficients"] = input_dict[key]["s_coefficients"][0]
            continue

        # adjust spline degree per case
        k_eff = min(k, n_points - 1)

        # Interpolate c
        c_spline = UnivariateSpline(
            input_dict[key]["z_l"],
            input_dict[key]["c_coefficients"],
            s=0.0,
            k=k_eff,
            ext=2,
        )
        t, c, order = c_spline._eval_args
        input_dict[key]["knots"] = list(t)
        input_dict[key]["order"] = int(order)
        input_dict[key]["c_coefficients"] = list(c)

        # Interpolate s (if present)
        if "s_coefficients" in input_dict[key]:
            s_spline = UnivariateSpline(
                input_dict[key]["z_l"],
                input_dict[key]["s_coefficients"],
                s=0.0,
                k=k_eff,
                ext=2,
            )
            t, c, order = s_spline._eval_args
            input_dict[key]["s_coefficients"] = c

        # After all the interpolation is done, remove z_l
        del input_dict[key]["z_l"]
