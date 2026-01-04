"""
Data visualization module.

This module provides auxiliary functions for data visualization.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import matplotlib.pyplot as plt

def set_metrics_plot():
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
