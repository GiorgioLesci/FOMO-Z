"""
Color domain estimation for FOMO-Z color cut parameter optimization.

This module provides auxiliary functions to assess optimal color domains.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import numpy as np


def c_domain(color, q_low=0.01, q_high=0.99):
    """
    Get the domain of `c`.

    Estimate the domain for the intercept parameter `c` in
    color-color conditions of the form:

        color > c
        color < c

    This function computes lower and upper quantile bounds to exclude extreme
    outliers while retaining the full range of relevant `c` values.

    Parameters
    ----------
    color : ndarray (1D)
        Input color array (may contain NaNs).
    q_low, q_high : float
        Lower and upper quantiles for robust domain estimation
        (0 < q_low < q_high < 1). Default: [0.01, 0.99].

    Returns
    -------
    c_min, c_max : float
        Lower and upper bounds for the valid c domain.
    """
    return np.quantile(color[np.isfinite(color)], [q_low, q_high])


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

def color_bin_edges(color_min, color_max, precision):
    """
    Generate evenly spaced bin edges for color values.
    
    Parameters
    ----------
    color_min : float
        Minimum color value.
    color_max : float
        Maximum color value.
    precision : float
        Bin width (spacing between edges).
    
    Returns
    -------
    numpy.ndarray
        Array of bin edges from color_min to color_max inclusive,
        spaced by precision.
    """
    return np.linspace(
        color_min,
        color_max,
        int((color_max - color_min) / precision) + 1,
    )
