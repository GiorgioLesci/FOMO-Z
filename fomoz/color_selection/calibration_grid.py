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
from .tools import (
    c_domain,
    color_bin_edges,
    compute_metrics,
    prob_target,
    get_color_color_mask,
)


class CalibrationGrid:
    """Calibrate grid color selections."""

    def __init__(
        self,
        dataset: Dataset,
        target_definition: TargetDefinition,
        thresholded_quantity_min,
        maximized_quantity="completeness",
        grid_precision=0.15,
        alpha=0.1,
        step=0.0001,
        do_bootstrap=True,
        n_bootstraps=1000,
    ):
        r"""
        Class calibrating optimal grid color selections.

        This class implements a general approach which identifies optimal selection regions
        in color-color spaces by directly exploring discretized grids. At each point of
        these color-color grids, the following probability to find a target
        object is computed:

        $P_T = \frac{N_T}{N_S}\,,$

        where $N_T$ is the number of target objects in the color-color bin, and
        $N_S$ is the total number of objects in the reference sample falling in the bin.

        For each limiting redshift value specified in `TargetDefinition`,
        this method searches for the set of cuts that
        ($\delta\mathcal{P}_{\rm step}$: `step`,
        $\mathcal{P}_{\rm min}$: `thresholded_quantity_min`,
        $\mathcal{C}$: `maximized_quantity`):
        - Update $\mathcal{P}_{\rm iter}^n = 1 - n\,\delta\mathcal{P}_{\rm step}$,
          where $n$ is the iteration index, starting from 0.
        - Remove the color-color bins where $P_T < \mathcal{P}_{\rm iter}^n$.
        - Compute the global purity $\mathcal{P}$, assuming that all color-color
          bins yield a significant measurement of $P_T$. If $\mathcal{P}>\mathcal{P}_{\rm min}$,
          repeat from step 1.
        - Mask not significant bins, based on the value of `alpha`. Then, recompute
          the selection metrics.

        Parameters
        ----------
        dataset : Dataset
            A `Dataset` object containing the dataset and band information.
        target_definition : TargetDefinition
            A `TargetDefinition` object containing target selections (e.g. redshift thresholds).
        thresholded_quantity_min : float or array-like
            Required minimum for the thresholded quantity, namely
            global purity (if `maximized_quantity == "completeness"`)
            or completeness (if `maximized_quantity == "purity"`) across all redshift bins.
            If float, the same threshold is applied at any limiting redshift point.
            If array-like, a different threshold is used at each limiting
            redshift.
        maximized_quantity : str, default="completeness"
            The quantity to maximize. Possibilities are either "completeness" or
            "purity".
        grid_precision : float, default=0.15
            Precision of the color-color grids. A smaller value yields a finer,
            more accurate grid at the cost of increased computation, RAM usage, and
            storage requirements.
        alpha : float, default=0.1
            Significance level used to discard statistically non-robust grid bins.
            For each bin, let k be the number of target objects (successes) out of
            n total objects (trials), and let p denote the true per-bin purity.
            Assuming a Binomial likelihood P(k | p, n) and a uniform prior on p
            over the interval [0, 1], the posterior distribution P(p | k, n)
            follows a Beta distribution.
            A bin is considered statistically significant only if the posterior
            probability that its true purity lies below the minimum acceptable
            purity p_min satisfies:

                P(p < p_min | k, n) < alpha.

            Therefore, alpha represents the maximum tolerated posterior probability
            that a bin's true purity is below the required threshold.
            The default value alpha = 0.1 corresponds to allowing at most a 10%
            posterior probability that the bin purity is below p_min.
            To disable this statistical filtering and retain all bins,
            set alpha = 1.
        step : float, default=0.0001
            Defines the threshold reduction on grid elements. Default is 0.0001.
        do_bootstrap : bool, default=True
            If True, derive uncertainties on C, P, and F through bootstrap.
        n_bootstraps : int, default=1000
            Number of bootstrap resamples to perform. It defines
            the number of resampling in error estimation.

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
        self.CDZs = dataset.CDZs
        self.PDZ_x = dataset.PDZ_x
        self.weights = np.array(self.dataset["weights"].values)
        self.color_spaces = dataset.color_spaces
        self.td = target_definition

        if isinstance(thresholded_quantity_min, float):
            self.thresholded_quantity_min = np.asarray(
                [thresholded_quantity_min] * len(self.td.z_lim)
            )
        else:
            if len(thresholded_quantity_min) != len(self.td.z_lim):
                raise ValueError(
                    f"thresholded_quantity_min length ({len(thresholded_quantity_min)}) "
                    f"must match z_lim length ({len(self.td.z_lim)})"
                )
            self.thresholded_quantity_min = np.asarray(thresholded_quantity_min)
        for t in self.thresholded_quantity_min:
            if not (0.0 <= t < 1.0):
                raise ValueError(
                    "thresholded_quantity_min accepted values lie in the range [0, 1)."
                )

        if maximized_quantity not in ["completeness", "purity"]:
            errMsg = "maximized_quantity must be either 'completeness' or 'purity'."
            raise ValueError(errMsg)
        if maximized_quantity == "completeness":
            self.thresholded_quantity = "purity"
        else:
            self.thresholded_quantity = "completeness"

        self.grid_precision = float(grid_precision)

        if alpha < 0 or alpha > 1:
            raise ValueError(
                f"alpha must range between 0 and 1. The current value is {alpha}."
            )
        self.alpha = alpha

        self.step = step
        self.do_bootstrap = do_bootstrap
        self.n_bootstraps = n_bootstraps

        # Initialize other attributes
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

        # Create the target masks
        self.target_mask = self.td._get_target_mask(
            self.dataset, self.td.z_lim, self.CDZs, self.PDZ_x
        )

        dataset_unweighted = self.dataset.copy()
        dataset_unweighted["weights"] = np.ones(len(self.dataset))
        self.target_mask_unweighted = self.td._get_target_mask(
            dataset_unweighted, self.td.z_lim, None, None
        )

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
        dictionary["metadata"][f"{self.thresholded_quantity}_min"] = [
            float(T) for T in self.thresholded_quantity_min
        ]
        dictionary["metadata"]["grid_precision"] = self.grid_precision

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
            value_bins = color_bin_edges(color_min, color_max, self.grid_precision)

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
            dictionary[cs]["mask"] = {}

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
            self.dict[cs]["mask"][i_z] = {}

            dict_stats[cs] = {}
            dict_stats[cs]["mask"] = {}
            dict_stats[cs]["color1_index"] = self.dict[cs]["color1_index"]
            dict_stats[cs]["color2_index"] = self.dict[cs]["color2_index"]
            dict_stats[cs]["valid"] = self.dict[cs]["valid"]

            # Probability to get target galaxies
            prob, N_good, N_total = prob_target(
                self.dict["metadata"],
                cs,
                self.dataset,
                self.target_mask[i_z],
                self.target_mask_unweighted[i_z],
                self.weights,
            )
            dict_stats[cs]["prob"] = prob
            dict_stats[cs]["N_good"] = N_good
            dict_stats[cs]["N_total"] = N_total

        # Remove bins with prob below a fixed threshold iteratively
        prob_min_bin = 1.0  # starting bin-level probability
        thresholded_quantity, P, C, F = 0.0, 0.0, 0.0, 0.0

        while prob_min_bin > 0.0:

            # Mask definition
            for cs in self.color_spaces:
                mask = dict_stats[cs]["prob"] >= prob_min_bin
                dict_stats[cs]["mask"][i_z] = mask

            mask = self.get_mask(i_z, dictionary=dict_stats)

            # Total completeness and purity
            TP, FP, FN, TN = compute_metrics(mask, self.target_mask[i_z], self.weights)
            C = TP / (TP + FN) if TP + FN > 0 else 0.0
            P = TP / (TP + FP) if TP + FP > 0 else 0.0

            # Check convergence
            if self.thresholded_quantity == "purity":
                thresholded_quantity = P
            elif self.thresholded_quantity == "completeness":
                thresholded_quantity = C
            else:
                raise ValueError(
                    f"Not allowed value for thresholded_quantity ({self.thresholded_quantity})."
                )

            if thresholded_quantity >= self.thresholded_quantity_min[i_z]:
                self.P[i_z] = P
                self.C[i_z] = C
                self.F[i_z] = FP / (FP + TN) if FP + TN > 0 else 0.0
                for cs in self.color_spaces:
                    self.dict[cs]["mask"][i_z] = dict_stats[cs]["mask"][i_z]
            else:
                break

            # Increase the bin-level threshold
            prob_min_bin -= self.step

        if prob_min_bin <= 0:
            errMsg = f"For z_l={self.td.z_lim[i_z]}, the convergence is attained for a value larger than the user-defined thresholded_quantity_min."
            errMsg += f" The current global P, C, and F are {self.P[i_z]}, {self.C[i_z]}, and {self.F[i_z]}, respectively."
            raise ValueError(errMsg)

        # Remove not robust bins, namely those with probability to have per-bin purity
        # p < prob_min_bin larger than alpha. Following the Bayes' theorem, we have
        # P(p | k, n) = P(k | p, n) P(p | n) / P(k | n),
        # where p is the per-bin purity, k is the number of successes, and n is the
        # number of trials. We assume that P(k | p, n) is a Binomial distribution and
        # P(p | n) = [0, 1] is a uniform prior.
        # Our selection criterion is P(p < p_min | k, n) < alpha.
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
                    significant = beta.cdf(prob_min_bin, k + 1, n - k + 1) < self.alpha
                    dict_significant[cs]["mask"][i, j] = significant

        # Re-define the masks and re-evaluate the metrics
        for cs in self.color_spaces:
            self.dict[cs]["mask"][i_z] = (
                dict_stats[cs]["mask"][i_z] & dict_significant[cs]["mask"]
            )

        mask = self.get_mask(i_z)
        TP, FP, FN, TN = compute_metrics(mask, self.target_mask[i_z], self.weights)
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

        for i in tqdm(range(self.n_bootstraps), desc="Deriving bootstrap errors"):

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

            target_mask = self.td._get_target_mask(
                bt_sample, self.td.z_lim, self.CDZs, self.PDZ_x
            )
            for i_z in range(len(self.td.z_lim)):
                mask = self.get_mask(i_z, dictionary=dict_temp)
                TP, FP, FN, TN = compute_metrics(mask, target_mask[i_z], self.weights)
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

        C_mean = C_stats[0] / self.n_bootstraps
        P_mean = P_stats[0] / self.n_bootstraps
        F_mean = F_stats[0] / self.n_bootstraps
        self.C_std = np.sqrt((C_stats[1] / self.n_bootstraps) - C_mean**2)
        self.P_std = np.sqrt((P_stats[1] / self.n_bootstraps) - P_mean**2)
        self.F_std = np.sqrt((F_stats[1] / self.n_bootstraps) - F_mean**2)

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

        for i_z in tqdm(range(len(self.td.z_lim)), desc="Looping over z_lim points"):
            self._process_for_one_z((i_z, store_iterations, cpu_number))

        # Compute bootstrap errors
        if self.do_bootstrap:
            self._estimate_errors()

        print(f"Converged!")
        print(f"P_tot = {self.P} +- {self.P_std}")
        print(f"C_tot = {self.C} +- {self.C_std}")
        print(f"F_tot = {self.F} +- {self.F_std}")

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

        return get_color_color_mask(zl_index, color_space, dictionary)

    def get_redshift_distribution(self, zl_index, bin_edges, normalize=True):
        """
        Get the redshift distribution of color-selected objects.

        Returns the redshift distribution relative to the input limiting redshift.
        If weights were provided to `Dataset`, these histograms are weighted accordingly.

        Parameters
        ------
        zl_index : int
            Index of the redshift threshold in TargetDefinition.z_lim.
        bin_edges : array-like
            Number of histogram bins.
        normalize : bool, optional
            If true, the output histogram is a PDF. Otherwise, it is a count histogram.

        Returns
        -------
        histogram : numpy.array
            Array of histogram values.
        """
        mask = self.get_mask(zl_index, color_space="all")
        dataset = self.dataset["redshift"][mask]
        hist, _ = np.histogram(
            dataset, bins=bin_edges, density=normalize, weights=self.weights[mask]
        )
        return hist
