"""
Derive optimal color selections based on color-color grids.

This module selects the optimal color-color grids,
given any limit on redshift and on other galaxy properties.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import copy
import multiprocessing as mp
import os
import time
from itertools import combinations

import numpy as np
import yaml
from tqdm import tqdm

from .tools import c_domain, color_bin_edges


class CalibrateGrid:
    """Calibrate grid color selections."""

    def __init__(self, data_model):
        r"""
        Class calibrating optimal grid color selections.

        This class...

        Attributes
        ----------
        dm : DataModel
            `DataModel` object.
        dict : dict
            Container for storing results based on color selections.
        P : numpy.array
            Total purity values.
        C : numpy.array
            Total completeness values.
        F : numpy.array
            Total failure rates.
        P_std : numpy.array
            Total purity error.
        C_std : numpy.array
            Total completeness error.
        F_std : numpy.array
            Total failure rate error.

        Parameters
        ----------
        data_model : DataModel
            A `DataModel` object containing the dataset, band information,
            redshift thresholds, and precomputed cut metrics.
        """
        self.dm = data_model

        self.dict_steps = {}
        self.P = np.zeros(len(self.dm.z_lim))
        self.C = np.zeros(len(self.dm.z_lim))
        self.F = np.zeros(len(self.dm.z_lim))
        self.C_std = np.full(len(self.dm.z_lim), np.nan)
        self.P_std = np.full(len(self.dm.z_lim), np.nan)
        self.F_std = np.full(len(self.dm.z_lim), np.nan)

        # Pre-computed color dictionary
        self.dict_colors = {}
        for color in self.dm.colors:
            self.dict_colors[color] = (
                self.dm.dataset[color[0]].values - self.dm.dataset[color[1]].values
            )

        # self.dict metadata
        self.dict = {}
        self.dict["metadata"] = {}
        self.dict["metadata"]["bands"] = self.dm.bands
        self.dict["metadata"]["z_l"] = [float(z) for z in self.dm.z_lim]
        self.dict["metadata"]["z_l_is_lower_limit"] = self.dm.is_zlim_lower
        self.dict["metadata"][f"{self.dm.thresholded_quantity}_min"] = [
            float(T) for T in self.dm.thresholded_quantity_min
        ]
        self.dict["metadata"]["grid_precision"] = self.dm.grid_precision

        # Color ranges and self.dict structure
        self.color_spaces = list(combinations(self.dm.colors, 2))
        for color_space in self.color_spaces:
            color1 = self.dict_colors[color_space[0]]
            color2 = self.dict_colors[color_space[1]]
            color1_min, color1_max = c_domain(color1, q_low=0.0001, q_high=0.9999)
            color2_min, color2_max = c_domain(color2, q_low=0.0001, q_high=0.9999)

            self.dict[color_space[0], color_space[1]] = {}
            self.dict[color_space[0], color_space[1]]["color1_limits"] = [
                color1_min,
                color1_max,
            ]
            self.dict[color_space[0], color_space[1]]["color2_limits"] = [
                color2_min,
                color2_max,
            ]

    def process(
        self,
        cpu_number=None,
        store_iterations=False,
    ):
        r"""
        Perform the iterative search for optimal color selection grids.

        Here, either completeness or purity are maximized, depending on what was
        set in the `DataModel` instance given in input to the constructor of this class.

        Parameters
        ----------
        cpu_number : int or None, optional
            Number of CPU cores to use for parallel processing. If None,
            defaults to using all available cores minus two.
        store_iterations : bool, optional
            If True, store the iteration steps in self.dict_steps.
            Default is False. For advanced users.

        Returns
        -------
        None
            Updates the following instance attributes in place:
            - ``self.dict`` : dict
                Nested dictionary of optimal iterative cuts for each
                redshift bin, restructured and interpolated.
            - ``self.P`` : numpy.ndarray
                Total purity across all bins.
            - ``self.C`` : numpy.ndarray
                Total completeness across all bins.
            - ``self.F`` : numpy.ndarray
                Total failure rate across all bins.
            - ``self.P_std`` : numpy.ndarray
                Total purity error across all bins.
            - ``self.C_std`` : numpy.ndarray
                Total completeness error across all bins.
            - ``self.F_std`` : numpy.ndarray
                Total failure rate error across all bins.
        """
        print("\nRunning the grid color selection calibration...")

        i_z = 0

        # Total number of good galaxies in the sample
        N_good_tot = np.count_nonzero(self.dm.target_mask[i_z])

        # Loop over color-color spaces to pre-compute statistics on the grids
        dict_stats = {}
        for color_space in self.color_spaces:

            # Initializations
            self.dict[color_space]["mask"] = {}
            self.dict[color_space]["mask"][i_z] = {}
            dict_stats[color_space] = {}

            # Grid points and total number of galaxies per bin
            color1_min, color1_max = self.dict[color_space]["color1_limits"]
            color2_min, color2_max = self.dict[color_space]["color2_limits"]
            color1_bins = color_bin_edges(
                color1_min, color1_max, self.dm.grid_precision
            )
            color2_bins = color_bin_edges(
                color2_min, color2_max, self.dm.grid_precision
            )

            color1 = self.dict_colors[color_space[0]]
            color2 = self.dict_colors[color_space[1]]
            dict_stats[color_space]["N_total"], _, _ = np.histogram2d(
                color1, color2, bins=[color1_bins, color2_bins]
            )

            # Per-bin number of galaxies fulfilling the input conditions
            # and per-bin probability to get good objects
            dict_stats[color_space]["N_good"] = np.histogram2d(
                color1[self.dm.target_mask[i_z]],
                color2[self.dm.target_mask[i_z]],
                bins=[color1_bins, color2_bins],
            )[0]

            prob = np.zeros_like(dict_stats[color_space]["N_good"])
            nonzero = dict_stats[color_space]["N_total"] > 0
            prob[nonzero] = (
                dict_stats[color_space]["N_good"][nonzero]
                / dict_stats[color_space]["N_total"][nonzero]
            )
            dict_stats[color_space]["prob"] = prob

        # Remove bins with prob below a fixed threshold iteratively
        prob_min_bin = 0.0  # starting bin-level probability
        thresholded_quantity, P, C = 0.0, 0.0, 0.0

        while thresholded_quantity < self.dm.thresholded_quantity_min[i_z]:

            N_good_sel, N_tot_sel = 0.0, 0.0
            for color_space in self.color_spaces:

                # Mask bins below the bin-level prob threshold
                mask = dict_stats[color_space]["prob"] >= prob_min_bin

                N_good_sel += dict_stats[color_space]["N_good"][mask].sum()
                N_tot_sel += dict_stats[color_space]["N_total"][mask].sum()

                dict_stats[color_space]["mask"] = mask

            # Total completeness
            if N_good_tot > 0:
                C = N_good_sel / N_good_tot

            # Total purity
            if N_tot_sel > 0:
                P = N_good_sel / N_tot_sel

            # Check convergence
            if self.dm.thresholded_quantity == "purity":
                thresholded_quantity = P
            elif self.dm.thresholded_quantity == "completeness":
                thresholded_quantity = C
            else:
                raise ValueError(
                    f"Not allowed value for thresholded_quantity ({self.dm.thresholded_quantity})."
                )

            if thresholded_quantity >= self.dm.thresholded_quantity_min:
                self.P[i_z] = P
                self.C[i_z] = C
                for color_space in self.color_spaces:
                    self.dict[color_space]["mask"][i_z] = dict_stats[color_space][
                        "mask"
                    ]
                break

            # Increase bin-level threshold
            prob_min_bin += self.dm.step

        del dict_stats

        print(f"Converged!")
        print(f"P_min_bin = {prob_min_bin:.3f}")
        print(f"P_tot     = {self.P[i_z]:.3f}")
        print(f"C_tot     = {self.C[i_z]:.3f}")

        print("Grid calibration complete.")

    def get_keys(self):
        """
        Get the color-color space keys.

        Returns the list of color-color space keys contained in `self.dict`.

        Returns
        ------
        keys : list of str
            List of color-color space keys.
        """
        return [key for key in list(self.dict.keys()) if key != "metadata"]

    def get_mask(self, zl_index, color_space="all"):
        """
        Get the calibration mask.

        Returns the calibration mask relative to the input limiting redshift.

        Parameters
        ------
        zl_index : int
            Index of the redshift threshold in DataModel.z_lim.
        color_space : str or tuple, optional
            It can be either "all" (all color-color spaces) or one of
            the available color-color spaces, in the form `((band1, band2), (band3, band4))`.
            Available color-color spaces can be obtained from `self.get_keys()`.

        Returns
        -------
        mask : numpy.array
            Array of booleans.
        """
        if color_space == "all":
            color_space = self.get_keys()
        else:
            if color_space not in self.get_keys():
                raise ValueError(f"Not allowed color_space value {color_space}.")
            color_space = [color_space]

        mask = np.full(len(self.dm.dataset), False)
        for cs in color_space:

            color1_min, color1_max = self.dict[cs]["color1_limits"]
            color2_min, color2_max = self.dict[cs]["color2_limits"]
            color1_bins = color_bin_edges(
                color1_min, color1_max, self.dm.grid_precision
            )
            color2_bins = color_bin_edges(
                color2_min, color2_max, self.dm.grid_precision
            )
            color1 = self.dict_colors[cs[0]]
            color2 = self.dict_colors[cs[1]]

            # Assign each galaxy to a bin
            i1 = np.digitize(color1, color1_bins) - 1
            i2 = np.digitize(color2, color2_bins) - 1

            # Guard against out-of-range points
            good_bins = self.dict[cs]["mask"][zl_index]
            valid = (
                (i1 >= 0)
                & (i1 < good_bins.shape[0])
                & (i2 >= 0)
                & (i2 < good_bins.shape[1])
            )

            # Select
            mask[valid] |= good_bins[i1[valid], i2[valid]]

        return mask
