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

import numpy as np
import yaml
from scipy.stats import beta
from tqdm import tqdm

from .dataset import Dataset
from .target_definition import TargetDefinition
from .tools import c_domain, color_bin_edges, compute_metrics, prob_target


class CalibrateGrid:
    """Calibrate grid color selections."""

    def __init__(self, dataset: Dataset, target_definition: TargetDefinition):
        r"""
        Class calibrating optimal grid color selections.

        This class... (also, talk about the threshold alpha!)

        Parameters
        ----------
        dataset : Dataset
            A `Dataset` object containing the dataset and band information.
        target_definition : TargetDefinition
            A `TargetDefinition` object containing target selections (e.g. redshift thresholds).

        Attributes
        ----------
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
        """
        self.dataset = dataset.dataset
        self.bands = dataset.bands
        self.colors = dataset.colors
        self.color_spaces = dataset.color_spaces

        self.td = target_definition

        self.dict_steps = {}
        self.P = np.zeros(len(self.td.z_lim))
        self.C = np.zeros(len(self.td.z_lim))
        self.F = np.zeros(len(self.td.z_lim))
        self.C_std = np.full(len(self.td.z_lim), np.nan)
        self.P_std = np.full(len(self.td.z_lim), np.nan)
        self.F_std = np.full(len(self.td.z_lim), np.nan)

        self.dict = {}
        self._update_dict_metadata(self.dict, self.dataset)
        self._update_dict(self.dict, self.dataset)

        # Create the target mask
        self.target_mask = self.td.get_target_mask(self.dataset, self.td.z_lim)

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
        dictionary["metadata"]["bands"] = self.bands
        dictionary["metadata"]["z_l"] = [float(z) for z in self.td.z_lim]
        dictionary["metadata"]["z_l_is_lower_limit"] = self.td.is_zlim_lower
        dictionary["metadata"][f"{self.td.thresholded_quantity}_min"] = [
            float(T) for T in self.td.thresholded_quantity_min
        ]
        dictionary["metadata"]["grid_precision"] = self.td.grid_precision

        dictionary["metadata"]["color_limits"] = {}
        for cs in self.color_spaces:
            value1 = dataset[cs[0][0]].values - dataset[cs[0][1]].values
            value2 = dataset[cs[1][0]].values - dataset[cs[1][1]].values
            color1_min, color1_max = c_domain(value1, q_low=0.0001, q_high=0.9999)
            color2_min, color2_max = c_domain(value2, q_low=0.0001, q_high=0.9999)

            # These are used to reconstruct the bin edges from the metadata
            dictionary["metadata"]["color_limits"][cs[0]] = [
                color1_min,
                color1_max,
            ]
            dictionary["metadata"]["color_limits"][cs[1]] = [
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
        for color in self.colors:

            value = dataset[color[0]].values - dataset[color[1]].values

            # keep the color-color grids fixed to the initial ones,
            # i.e. those in self.dict
            color_min, color_max = self.dict["metadata"]["color_limits"][color]
            value_bins = color_bin_edges(color_min, color_max, self.td.grid_precision)

            index = np.digitize(value, value_bins) - 1
            nbins = len(value_bins) - 1
            valid = (index >= 0) & (index < nbins)

            dict_colors[color] = {"index": index, "valid": valid}

        # Update the valid color indices in dictionary
        for cs in self.color_spaces:
            color1 = cs[0]
            color2 = cs[1]
            valid = dict_colors[color1]["valid"] & dict_colors[color2]["valid"]
            i1 = dict_colors[color1]["index"][valid]
            i2 = dict_colors[color2]["index"][valid]

            dictionary[cs] = {}
            dictionary[cs]["valid"] = valid
            dictionary[cs]["color1_index"] = i1
            dictionary[cs]["color2_index"] = i2

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

        # Loop over color-color spaces to pre-compute statistics on the grids
        dict_stats = {}  # temporary dict containing grid statistics and masks
        for cs in self.color_spaces:

            # Dictionary initializations
            self.dict[cs]["mask"] = {}

            dict_stats[cs] = {}
            dict_stats[cs]["mask"] = {}
            dict_stats[cs]["color1_index"] = self.dict[cs]["color1_index"]
            dict_stats[cs]["color2_index"] = self.dict[cs]["color2_index"]
            dict_stats[cs]["valid"] = self.dict[cs]["valid"]

            # Probability to get target galaxies
            prob, N_good, N_total = prob_target(
                self.dict["metadata"], cs, self.dataset, self.target_mask[i_z]
            )
            dict_stats[cs]["prob"] = prob
            dict_stats[cs]["N_good"] = N_good
            dict_stats[cs]["N_total"] = N_total

        # Remove bins with prob below a fixed threshold iteratively
        prob_min_bin = 1.0  # starting bin-level probability
        thresholded_quantity, P, C, F = 0.0, 0.0, 0.0, 0.0

        while prob_min_bin > 0.0:

            # Mask bins below the bin-level prob threshold
            for cs in self.color_spaces:
                mask = dict_stats[cs]["prob"] >= prob_min_bin
                dict_stats[cs]["mask"][i_z] = mask

            mask = self.get_mask(i_z, dictionary=dict_stats)

            # Total completeness and purity
            TP, FP, FN, TN = compute_metrics(mask, self.target_mask[i_z])
            C = TP / (TP + FN) if TP + FN > 0 else 0.0
            P = TP / (TP + FP) if TP + FP > 0 else 0.0

            # Check convergence
            if self.td.thresholded_quantity == "purity":
                thresholded_quantity = P
            elif self.td.thresholded_quantity == "completeness":
                thresholded_quantity = C
            else:
                raise ValueError(
                    f"Not allowed value for thresholded_quantity ({self.td.thresholded_quantity})."
                )

            if thresholded_quantity >= self.td.thresholded_quantity_min[i_z]:
                self.P[i_z] = P
                self.C[i_z] = C
                self.F[i_z] = FP / (FP + TN) if FP + TN > 0 else 0.0
                for cs in self.color_spaces:
                    self.dict[cs]["mask"][i_z] = dict_stats[cs]["mask"][i_z]
            else:
                break

            # Increase the bin-level threshold
            prob_min_bin -= self.td.step

        if prob_min_bin <= 0:
            raise ValueError(
                f"Could not find a calibration for the redshift value {self.td.z_lim[i_z]}."
            )

        # Remove not robust bins, namely those with probability to have per-bin purity
        # p < prob_min_bin larger than alpha. Following the Bayes' theorem, we have
        # P(p | k, n) = P(k | p, n) P(p | n) / P(k | n),
        # where p is the per-bin purity, k is the number of successes, and n is the
        # number of trials. We assume that P(k | p, n) is a Binomial distribution and
        # P(p | n) = [0, 1] is a uniform prior.
        # Our selection criterion is P(p < p_min | k, n) < alpha.

        if self.td.alpha < 1.0:

            dict_significant = {}

            for cs in self.color_spaces:

                dict_significant[cs] = {}
                dict_significant[cs]["mask"] = {}
                shape = dict_stats[cs]["prob"].shape
                dict_significant[cs]["mask"] = np.zeros(shape, dtype=bool)

                for i in range(shape[0]):
                    for j in range(shape[1]):
                        k = dict_stats[cs]["N_good"][i][j]
                        n = dict_stats[cs]["N_total"][i][j]
                        if n == 0:
                            continue
                        # Given the observed data (k successes out of n), compute the posterior
                        # probability that (true purity < prob_min_bin) is smaller than alpha.
                        significant = (
                            beta.cdf(prob_min_bin, k + 1, n - k + 1) < self.td.alpha
                        )
                        dict_significant[cs]["mask"][i, j] = significant

            # Re-define the masks and re-evaluate the metrics
            for cs in self.color_spaces:
                self.dict[cs]["mask"][i_z] = (
                    dict_stats[cs]["mask"][i_z] & dict_significant[cs]["mask"]
                )

            mask = self.get_mask(i_z)
            TP, FP, FN, TN = compute_metrics(mask, self.target_mask[i_z])
            self.P[i_z] = TP / (TP + FP) if TP + FP > 0 else 0.0
            self.C[i_z] = TP / (TP + FN) if TP + FN > 0 else 0.0
            self.F[i_z] = FP / (FP + TN) if FP + TN > 0 else 0.0

    def _estimate_errors(self):
        """
        Estimate bootstrap errors.

        This method computes mean and standard deviation of completeness, purity,
        and failure rate, based on bootstrap resampling. Results are
        stored as instance attributes:
        - C_std: Standard deviation of completeness
        - P_std: Standard deviation of purity
        - F_std: Standard deviation of failure rate

        Raises
        ------
        ValueError
            If calibration has not been run prior to error estimation.
        """
        if np.sum(self.C) == 0.0:
            raise ValueError("No calibration is available! Run process() first.")

        C_stats = np.zeros((2, len(self.td.z_lim)))  # [sum, sum_of_squares]
        P_stats = np.zeros((2, len(self.td.z_lim)))
        F_stats = np.zeros((2, len(self.td.z_lim)))

        for i in tqdm(range(self.td.n_bootstraps), desc="Deriving bootstrap errors"):

            # Bootstrap sample
            indices = np.random.choice(
                len(self.dataset), size=len(self.dataset), replace=True
            )
            bt_sample = self.dataset.iloc[indices]

            # Pre-compute colors
            dict_temp = {}
            self._update_dict(dict_temp, bt_sample)

            # Set the color-color masks equal to the calibrated ones
            for cs in self.color_spaces:
                dict_temp[cs]["mask"] = {}
                for i_z in range(len(self.td.z_lim)):
                    dict_temp[cs]["mask"][i_z] = self.dict[cs]["mask"][i_z]

            # Compute the metrics
            C_temp = np.zeros(len(self.td.z_lim))
            P_temp = np.zeros(len(self.td.z_lim))
            F_temp = np.zeros(len(self.td.z_lim))

            target_mask = self.td.get_target_mask(bt_sample, self.td.z_lim)
            for i_z in range(len(self.td.z_lim)):
                mask = self.get_mask(i_z, dictionary=dict_temp)
                TP, FP, FN, TN = compute_metrics(mask, target_mask[i_z])
                C_temp[i_z] = TP / (TP + FN) if TP + FN > 0 else 0.0
                P_temp[i_z] = TP / (TP + FP) if TP + FP > 0 else 0.0
                F_temp[i_z] = FP / (FP + TN) if FP + TN > 0 else 0.0

            C_stats[0] += C_temp
            C_stats[1] += C_temp**2.0
            P_stats[0] += P_temp
            P_stats[1] += P_temp**2.0
            F_stats[0] += F_temp
            F_stats[1] += F_temp**2.0

            del bt_sample, C_temp, P_temp, F_temp

        C_mean = C_stats[0] / self.td.n_bootstraps
        P_mean = P_stats[0] / self.td.n_bootstraps
        F_mean = F_stats[0] / self.td.n_bootstraps
        self.C_std = np.sqrt((C_stats[1] / self.td.n_bootstraps) - C_mean**2)
        self.P_std = np.sqrt((P_stats[1] / self.td.n_bootstraps) - P_mean**2)
        self.F_std = np.sqrt((F_stats[1] / self.td.n_bootstraps) - F_mean**2)

    def process(
        self,
        cpu_number=None,
        store_iterations=False,
    ):
        r"""
        Perform the iterative search for optimal color selection grids.

        Here, either completeness or purity are maximized, depending on what was
        set in the `TargetDefinition` instance given in input to the constructor of this class.

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
            for i_z in tqdm(range(len(self.td.z_lim)), desc="Looping over z_lim points")
        ]

        # Compute bootstrap errors
        if self.td.do_bootstrap:
            self._estimate_errors()

        print(f"Converged!")
        print(f"P_tot     = {self.P} +- {self.P_std}")
        print(f"C_tot     = {self.C} +- {self.C_std}")
        print(f"F_tot     = {self.F} +- {self.F_std}")

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
            Index of the redshift threshold in TargetDefinition.z_lim.
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
