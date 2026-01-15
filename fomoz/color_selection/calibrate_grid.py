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
        self.color_spaces = list(combinations(self.dm.colors, 2))

        self.dict_steps = {}
        self.P = np.zeros(len(self.dm.z_lim))
        self.C = np.zeros(len(self.dm.z_lim))
        self.F = np.zeros(len(self.dm.z_lim))
        self.C_std = np.full(len(self.dm.z_lim), np.nan)
        self.P_std = np.full(len(self.dm.z_lim), np.nan)
        self.F_std = np.full(len(self.dm.z_lim), np.nan)

        self.dict = {}
        self._update_dict_metadata(self.dict, self.dm.dataset)
        self._update_dict(self.dict, self.dm.dataset)

    def _update_dict_metadata(self, dictionary, dataset):
        r"""
        Update the input dictionary's metadata, given the input data set.

        Parameters
        ----------
        dictionary : dict
            Dictionary.
        dataset : pandas.DataFrame
            Input dataset containing photometric bands.
        """
        dictionary["metadata"] = {}
        dictionary["metadata"]["bands"] = self.dm.bands
        dictionary["metadata"]["z_l"] = [float(z) for z in self.dm.z_lim]
        dictionary["metadata"]["z_l_is_lower_limit"] = self.dm.is_zlim_lower
        dictionary["metadata"][f"{self.dm.thresholded_quantity}_min"] = [
            float(T) for T in self.dm.thresholded_quantity_min
        ]
        dictionary["metadata"]["grid_precision"] = self.dm.grid_precision

        dictionary["metadata"]["color_limits"] = {}
        for color_space in self.color_spaces:
            value1 = (
                dataset[color_space[0][0]].values - dataset[color_space[0][1]].values
            )
            value2 = (
                dataset[color_space[1][0]].values - dataset[color_space[1][1]].values
            )
            color1_min, color1_max = c_domain(value1, q_low=0.0001, q_high=0.9999)
            color2_min, color2_max = c_domain(value2, q_low=0.0001, q_high=0.9999)

            # These are used to reconstruct the bin edges from the metadata
            dictionary["metadata"]["color_limits"][color_space[0]] = [
                color1_min,
                color1_max,
            ]
            dictionary["metadata"]["color_limits"][color_space[1]] = [
                color2_min,
                color2_max,
            ]

    def _update_dict(self, dictionary, dataset):
        r"""
        Update the input dictionary, given the input data set.

        Parameters
        ----------
        dictionary : dict
            Dictionary.
        dataset : pandas.DataFrame
            Input dataset containing photometric bands.
        """
        # Color values and object index dictionary
        dict_colors = {}
        for color in self.dm.colors:

            value = dataset[color[0]].values - dataset[color[1]].values

            # keep the color-color grids fixed to the initial ones!
            color_min, color_max = dictionary["metadata"]["color_limits"][color]
            value_bins = color_bin_edges(color_min, color_max, self.dm.grid_precision)

            index = np.digitize(value, value_bins) - 1
            nbins = len(value_bins) - 1
            valid = (index >= 0) & (index < nbins)

            dict_colors[color] = {"index": index, "valid": valid}

        # Update the valid color indices in dictionary
        for color_space in self.color_spaces:
            color1 = color_space[0]
            color2 = color_space[1]
            valid = dict_colors[color1]["valid"] & dict_colors[color2]["valid"]
            i1 = dict_colors[color1]["index"][valid]
            i2 = dict_colors[color2]["index"][valid]

            dictionary[color_space] = {}
            dictionary[color_space]["valid"] = valid
            dictionary[color_space]["color1_index"] = i1
            dictionary[color_space]["color2_index"] = i2

    def _process_for_one_z(self, args):
        r"""
        Calibrate the color cut selection on grids for a single redshift threshold `z_l`.

        See the documentation of the class constructor for a detailed description of the
        iterative process implemented in this function.

        Parameters
        ----------
        args : tuple
            A 3-element tuple containing:
            - i_z : float
                The index of the redshift threshold being processed.
            - store_iterations : bool
                If True, store the iteration steps in self.dict_steps.
            - cpu_number : int
                Number of CPU cores to use for parallel processing.
        """
        i_z, store_iterations, cpu_number = args

        # Total number of good galaxies in the sample
        N_good_tot = np.count_nonzero(self.dm.target_mask[i_z])

        # Loop over color-color spaces to pre-compute statistics on the grids
        dict_stats = {}  # temporary dict containing grid statistics and masks
        for color_space in self.color_spaces:

            # Dictionary initializations
            self.dict[color_space]["mask"] = {}

            dict_stats[color_space] = {}
            dict_stats[color_space]["mask"] = {}
            dict_stats[color_space]["color1_index"] = self.dict[color_space][
                "color1_index"
            ]
            dict_stats[color_space]["color2_index"] = self.dict[color_space][
                "color2_index"
            ]
            dict_stats[color_space]["valid"] = self.dict[color_space]["valid"]

            # Grid points and total number of galaxies per bin
            color1_min, color1_max = self.dict["metadata"]["color_limits"][
                color_space[0]
            ]
            color2_min, color2_max = self.dict["metadata"]["color_limits"][
                color_space[1]
            ]

            color1_bins = color_bin_edges(
                color1_min, color1_max, self.dm.grid_precision
            )
            color2_bins = color_bin_edges(
                color2_min, color2_max, self.dm.grid_precision
            )

            color1 = (
                self.dm.dataset[color_space[0][0]].values
                - self.dm.dataset[color_space[0][1]].values
            )
            color2 = (
                self.dm.dataset[color_space[1][0]].values
                - self.dm.dataset[color_space[1][1]].values
            )
            N_total, _, _ = np.histogram2d(
                color1, color2, bins=[color1_bins, color2_bins]
            )

            # Per-bin number of galaxies fulfilling the input conditions
            # and per-bin probability to get good objects
            N_good = np.histogram2d(
                color1[self.dm.target_mask[i_z]],
                color2[self.dm.target_mask[i_z]],
                bins=[color1_bins, color2_bins],
            )[0]

            prob = np.zeros_like(N_good)
            nonzero = N_total > 0
            prob[nonzero] = N_good[nonzero] / N_total[nonzero]
            dict_stats[color_space]["prob"] = prob

        # Remove bins with prob below a fixed threshold iteratively
        prob_min_bin = 0.0  # starting bin-level probability
        thresholded_quantity, P, C = 0.0, 0.0, 0.0

        while thresholded_quantity < self.dm.thresholded_quantity_min[i_z]:

            # Mask bins below the bin-level prob threshold
            for color_space in self.color_spaces:
                mask = dict_stats[color_space]["prob"] >= prob_min_bin
                dict_stats[color_space]["mask"][i_z] = mask

            mask = self.get_mask(i_z, dictionary=dict_stats)
            N_good_sel = np.count_nonzero(mask & self.dm.target_mask[i_z])
            N_tot_sel = np.count_nonzero(mask)

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

            if thresholded_quantity >= self.dm.thresholded_quantity_min[i_z]:
                self.P[i_z] = P
                self.C[i_z] = C
                for color_space in self.color_spaces:
                    self.dict[color_space]["mask"][i_z] = dict_stats[color_space][
                        "mask"
                    ][i_z]
                break

            # Increase bin-level threshold
            prob_min_bin += self.dm.step

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

        results = [
            self._process_for_one_z((i_z, store_iterations, cpu_number))
            for i_z in tqdm(range(len(self.dm.z_lim)), desc="Looping over z_lim points")
        ]

        print(f"Converged!")
        print(f"P_tot     = {self.P}")
        print(f"C_tot     = {self.C}")

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

    def get_mask(self, zl_index, color_space="all", dictionary=None):
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
        dictionary : dict, optional
            The color-color space dictionary, containing grid masks. If not provided,
            the internal dictionary (`self.dict`) is used.

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

        if dictionary is None:
            dictionary = self.dict

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
