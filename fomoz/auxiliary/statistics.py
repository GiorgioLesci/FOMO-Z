"""
Compute interpolations, true/false positives, and true/false negatives.

This module provides auxiliary functions for the computation of
color selection metrics and interpolations.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

from .color_domain import color_bin_edges


def compute_metrics(mask, target_mask, weights):
    """
    Compute classification metrics (TP, FP, FN, TN).

    This function computes true/false positives and true/false negatives
    based on the input masks.

    Parameters
    ----------
    mask : array-like of bool
        Boolean mask of selected objects by color cuts.
    target_mask : np.ndarray
        Mask of target object probabilities (i.e., probability to
        satisfy redshift selections + additional rules), multiplied
        by `weights`.
    weights : np.ndarray
        Per-object weights.

    Returns
    -------
    TP : float
        True Positives.
    FP : float
        False Positives.
    FN : float
        False Negatives.
    TN : float
        True Negatives.
    """
    if (
        np.issubdtype(target_mask.dtype, np.bool_)
        or np.issubdtype(target_mask.dtype, np.int8)
    ) and np.allclose(weights, 1.0):
        sel = mask
        tgt = target_mask

        TP = np.count_nonzero(sel & tgt)
        FP = np.count_nonzero(sel) - TP
        FN = np.count_nonzero(tgt) - TP
        TN = len(mask) - TP - FP - FN

    else:
        TP = target_mask[mask].sum()
        FN = target_mask[~mask].sum()

        n_sel = weights[mask].sum()
        n_unsel = weights[~mask].sum()

        FP = n_sel - TP
        TN = n_unsel - FN

    return TP, FP, FN, TN


def compute_for_simple(args):
    """
    Compute classification metrics for a simple threshold on a single color.

    This function applies a threshold `c` on a color array,
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
        - weights : np.ndarray
            Prior weights.

    Returns
    -------
    list of int
        Concatenated list of metrics in the following order:
        [TP_up, FP_up, FN_up, TN_up, TP_low, FP_low, FN_low, TN_low]
        where the suffix `_up` refers to the mask `color > c` and `_low`
        refers to `color < c`.
    """
    c, color, target_mask, weights = args
    mask_up = color > c
    mask_low = color < c

    return [
        *compute_metrics(mask_up, target_mask, weights),
        *compute_metrics(mask_low, target_mask, weights),
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
        - weights : np.ndarray
            Prior weights.

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
    s, c, color1, color2, target_mask, weights = args

    cuts = [
        color1 > s * color2 + c,
        color1 > -s * color2 + c,
        color1 < s * color2 + c,
        color1 < -s * color2 + c,
    ]

    return [compute_metrics(mask, target_mask, weights) for mask in cuts]


def combine_optimal_cuts(
    z_lim, color_cache, target_mask, weights, dictionary, C_tot, P_tot, F_tot
):
    """
    Combine optimal color cuts and compute global metrics.

    This method aggregates purity, completeness, and
    failure rates across a set of optimal cuts stored in
    `dictionary`. The combined results are accumulated into the
    provided arrays `C_tot`, `P_tot`, and `F_tot`.

    Parameters
    ----------
    color_cache : dict
        Dictionary mapping color pairs (mag1, mag2) to precomputed
        color arrays (mag1 - mag2). Keys are tuples of band names,
        values are numpy arrays of color values.
    target_mask : np.ndarray
        Boolean mask of target objects (satisfying redshift + additional rules).
    weights : np.ndarray
        Prior weights.
    dictionary : dict
        Dictionary of selected optimal cuts,
        where each entry contains the metrics for a cut.
    C_tot : numpy.ndarray
        Array that accumulates total completeness values across cuts.
    P_tot : numpy.ndarray
        Array that accumulates total purity values across cuts.
    F_tot : numpy.ndarray
        Array that accumulates total failure rate values across cuts.

    Returns
    -------
    None
        Updates the input arrays (`C_tot`, `P_tot`, `F_tot`) in place
        but does not return a value.
    """
    n_galaxies = len(target_mask[0])

    for i, z_l in enumerate(z_lim):

        mask_total = np.zeros(n_galaxies, dtype=bool)
        cuts_for_zl = dictionary.get(z_l, {})

        for base_pair, comp_cuts in cuts_for_zl.items():
            mag1, mag2 = base_pair
            color1 = color_cache[(mag1, mag2)]

            for comp_pair, conditions in comp_cuts.items():
                if comp_pair == base_pair:
                    for cond in conditions:
                        c = cond["c"]
                        if cond["condition_index"] == 0:
                            mask = color1 > c
                        else:
                            mask = color1 < c
                        mask_total |= mask
                else:
                    mag3, mag4 = comp_pair
                    color2 = color_cache[(mag3, mag4)]

                    for cond in conditions:
                        s = cond["s"]
                        c = cond["c"]
                        idx = cond["condition_index"]

                        if idx == 0:
                            mask = color1 > s * color2 + c
                        elif idx == 1:
                            mask = color1 > -s * color2 + c
                        elif idx == 2:
                            mask = color1 < s * color2 + c
                        else:
                            mask = color1 < -s * color2 + c

                        mask_total |= mask

        TP, FP, FN, TN = compute_metrics(mask_total, target_mask[i], weights)

        C_tot[i] = TP / (TP + FN) if TP + FN > 0 else 0.0
        P_tot[i] = TP / (TP + FP) if TP + FP > 0 else 0.0
        F_tot[i] = FP / (FP + TN) if FP + TN > 0 else 0.0


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


def distribution_contours(x, y, nbins, levels):
    """
    Compute 2D density contours for a given distribution of points.

    This function creates a 2D histogram of (x, y) data and computes
    density thresholds corresponding to specified probability levels.
    The contours represent iso-density lines that enclose the given
    percentages of the total probability mass.

    Parameters
    ----------
    x : array-like
        1D array of x-coordinates of data points.
    y : array-like
        1D array of y-coordinates of data points. Must have same length as x.
    nbins : int or tuple of int
        Number of bins for the 2D histogram. If int, same number of bins
        for both dimensions. If tuple, (nxbins, nybins).
    levels : array-like
        Probability levels for which to compute contours, expressed as
        fractions between 0 and 1 (e.g., [0.68, 0.95] for 68% and 95%
        confidence intervals).

    Returns
    -------
    X : ndarray
        2D array of x-coordinates for grid points (shape: (nbins, nbins)).
    Y : ndarray
        2D array of y-coordinates for grid points (shape: (nbins, nbins)).
    H_norm : ndarray
        2D array of normalized histogram values representing probability
        density (shape: (nbins, nbins)). Values sum to 1.
    level_density : list
        Density thresholds corresponding to each input level, sorted from
        highest to lowest probability (e.g., [density_95, density_68] for
        input levels [0.68, 0.95]). These can be used as `levels` parameter
        in matplotlib's `contour` function.
    """
    x_valid = np.isfinite(x)
    y_valid = np.isfinite(y)
    valid_mask = x_valid & y_valid

    H, xedges, yedges = np.histogram2d(x[valid_mask], y[valid_mask], bins=nbins)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters)

    H_norm = H / H.sum()
    sorted_vals = np.sort(H_norm.ravel())[::-1]
    cumsum = np.cumsum(sorted_vals)

    level_density = [
        sorted_vals[np.searchsorted(cumsum, level)] for level in np.sort(levels)[::-1]
    ]

    return X, Y, H_norm, level_density


def prob_target(metadata, cs, dataset, target_mask, weights):
    """
    Compute the per-bin probability of selecting target objects in a 2D color–color grid.

    Parameters
    ----------
    cs : tuple
        Color–color space definition in the form
        ((band1, band2), (band3, band4)).
    metadata : dict
        Dictionary containing grid metadata, including color limits and grid precision.
    dataset : pandas.DataFrame
        Input dataset containing the photometric bands used to compute colors.
    target_mask : numpy.ndarray of bool or floats
        Float (possibly weighted) mask selecting target objects (e.g. high-redshift galaxies).
    weights : numpy.array of floats
        Prior weights.

    Returns
    -------
    prob : numpy.ndarray
        2D array of per-bin probabilities, defined as the ratio of target objects
        to total objects in each color–color bin.
    n_successes : numpy.ndarray
        2D array.
    n_trials : numpy.ndarray
        2D array of per-bin total number objects.
    """
    color1_min, color1_max = metadata["color_limits"][cs[0]]
    color2_min, color2_max = metadata["color_limits"][cs[1]]

    color1_bins = color_bin_edges(color1_min, color1_max, metadata["grid_precision"])
    color2_bins = color_bin_edges(color2_min, color2_max, metadata["grid_precision"])

    color1 = dataset[cs[0][0]].values - dataset[cs[0][1]].values
    color2 = dataset[cs[1][0]].values - dataset[cs[1][1]].values

    n_total, _, _ = np.histogram2d(
        color1, color2, bins=[color1_bins, color2_bins], weights=weights
    )

    # n_trials
    n_trials_denominator, _, _ = np.histogram2d(
        color1, color2, bins=[color1_bins, color2_bins], weights=weights**2.0
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        n_trials = np.divide(
            n_total * n_total,
            n_trials_denominator,
            out=np.zeros_like(n_total),
            where=n_trials_denominator > 0,
        )

    # Per-bin number of galaxies fulfilling the input conditions
    # and per-bin probability to get good objects
    n_good = np.histogram2d(
        color1,
        color2,
        bins=[color1_bins, color2_bins],
        weights=target_mask.astype(float),
    )[0]

    prob = np.zeros_like(n_good, dtype=float)
    nonzero = n_total > 0
    prob[nonzero] = n_good[nonzero] / n_total[nonzero]

    return prob, prob * n_trials, n_trials


def get_color_color_mask(zl_index, color_space, dictionary):
    """
    Get the color-color grid calibration mask.

    Returns the calibration mask relative to the input limiting redshift.

    Parameters
    ------
    zl_index : int
        Index of the redshift threshold in TargetDefinition.z_lim.
    color_space : str or tuple
        Set of color spaces.
    dictionary : dict
        The color-color space dictionary, containing grid masks.

    Returns
    -------
    mask : numpy.array
        Array of booleans.
    """
    N_objects = len(dictionary[color_space[0]]["valid"])
    mask = np.full(N_objects, False)
    for cs in color_space:

        # Assign each galaxy to a bin.
        # Here, out-of-grid points are excluded.
        i1 = dictionary[cs]["color1_index"]
        i2 = dictionary[cs]["color2_index"]

        # Select
        valid = dictionary[cs]["valid"]
        mask[valid] |= dictionary[cs]["mask"][zl_index][i1, i2]

    return mask
