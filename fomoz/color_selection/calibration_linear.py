"""
Derive optimal color selections based on linear cuts.

This module selects the optimal color conditions and the corresponding
optimal parameters, given any limit on redshift and on other galaxy properties.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import copy
import multiprocessing as mp
import os
import pickle
import warnings

import numexpr as ne
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .dataset import Dataset
from .target_definition import TargetDefinition
from .tools import (
    MeasureLinear,
    build_nested_dict,
    combine_optimal_cuts,
    float_representer,
    inline_list_representer,
    interpolate_parameters,
    make_yaml_serializable,
    restructure_dict,
)

# Register the custom representers for both default and safe dumpers
yaml.add_representer(float, float_representer)
yaml.add_representer(float, float_representer, Dumper=yaml.SafeDumper)

yaml.add_representer(list, inline_list_representer)
yaml.add_representer(list, inline_list_representer, Dumper=yaml.SafeDumper)


class CalibrationLinear:
    """Calibrate linear color selections."""

    def __init__(
        self,
        dataset: Dataset,
        target_definition: TargetDefinition,
        thresholded_quantity_min,
        maximized_quantity="completeness",
        maximized_quantity_min=0.05,
        threshold_buffer=0.05,
        s_samples=np.linspace(0.3, 2.5, 50),
        c_precision=0.05,
        k=1,
        step=0.0001,
        do_bootstrap=True,
        n_bootstraps=1000,
    ):
        r"""
        Class calibrating optimal linear color selections.

        This class implements methods that aim to select objects that occupy the
        outskirts of color-color distributions, a characteristic feature of, e.g.,
        background galaxy populations. For other science cases, refer to the
        `CalibrationGrid` class.


        The following color conditions are considered in each color-color space:

          - color1 > c1
          - color1 < c2
          - color2 > c3
          - color2 < c4
          - color1 > s1 * color2 + c5
          - color1 > s2 * color2 + c6
          - color1 < s3 * color2 + c7
          - color1 < s4 * color2 + c8

        where c1...c8 are real numbers, s1 and s3 are non-zero positive real numbers,
        while s2 and s4 are non-zero negative real numbers. All color conditions,
        within the same space and between different spaces, are connected
        through the logical "or" operator.

        The maximized quantity (either completeness or purity) is defined by the
        `maximized_quantity` value given in input to this constructor.
        The method implemented here incrementally relaxes the single color cut purity (completeness)
        threshold in order to identify a set of cuts that maximize total completeness
        (purity) while still satisfying the global threshold. The algorithm
        proceeds iteratively, starting from a strict initial threshold and reducing
        it by `step` until the total purity (or completeness) remains above
        `thresholded_quantity_min`.
        Also bootstrap errors on the metrics are derived.

        For each limiting redshift value specified in `TargetDefinition`,
        this method searches for the set of cuts that
        ($\delta\mathcal{P}_{\rm step}$: `step`,
        $\mathcal{C}_{\rm min}$: `maximized_quantity_min`,
        $\mathcal{P}_{\rm min}$: `thresholded_quantity_min`):
        - Update $\mathcal{P}_{\rm iter}^n = 1 - n\,\delta\mathcal{P}_{\rm step}$,
          where $n$ is the iteration index, starting from 0.
        - For any $(x,y)$ color pair, select the $k$th condition
          if there exists at least one parameter set that yields
          a single-condition purity $\mathcal{P}^k>\mathcal{P}_{\rm iter}^n$
          and a single-condition completeness $\mathcal{C}^k>\mathcal{C_min}$.
        - For each color condition selected on step 2, define the optimal
          parameter set as the one that maximizes $\mathcal{C}^k$.
        - Combine the selected conditions, with optimal parameters,
          in all $(x,y)$ spaces through the $\lor$ logical operator
          Then, evaluate the global purity, $\mathcal{P}$.
        - Repeat from point 1 as long as
          $\mathcal{P}(z_{\rm l})>\mathcal{P}_{\rm min}$.

        After the optimal cuts are found for all redshift bins, the results are:
        1. Combined into global purity, completeness, and failure rate metrics.
        2. Restructured into an expression-based dictionary.
        3. Interpolated across redshift bins to provide continuous cut parameters.

        Single color conditions having failure rate equal to 1 are always excluded.

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
        maximized_quantity_min : float, default=0.05
            Required minimum for the maximized completeness (if `maximized_quantity == "completeness"`)
            or purity (if `maximized_quantity == "purity"`) across all redshift bins.
            This is not a threshold on the global value of the maximized quantity.
            Only single color conditions that meet the requirement on
            this parameter are considered. The default
            value is 0.05, namely 5%. This threshold is useful to exclude random fluctuations
            and reduce statistical uncertainties on global purity and completeness.
        threshold_buffer : float, default=0.05
            Quantity subtracted from `thresholded_quantity_min`.
            Color conditions not satisfying the final threshold, namely
            `thresholded_quantity_min - threshold_buffer`, are not
            considered in the optimization process. This is useful to avoid filling memory
            with unused information. The default value is 0.05, namely a 5 percentage point
            subtraction. If this value is too small, the optimization method raises an error.
        s_samples : array-like, default=np.linspace(0.3, 1.7, 30)
            Array of slope values, s_j, for generating linear cuts in color-color space.
            These samples must be strictly positive.
        c_precision : float, default=0.05
            Precision of the c_i parameters sampling. Smaller values
            increase accuracy but significantly increase the number of operations.
            If `c_precision` is halved, expect about 2 times more c_i steps per s_j
            (also runtime and RAM roughly double).
        k : int, default=1
            Requested degree of the spline interpolation, used to interpolate color
            condition parameters. Default is 1 (linear).
            The actual degree is adjusted case by case:
            - If there are fewer than ``k+1`` points, the degree is reduced
              to ``n_points - 1``.
            - If there is only one point, interpolation is skipped.
        step : float, default=0.0001
            Defines the threshold reduction on single color conditions. Default is 0.0001.
        do_bootstrap : bool, default=True
            If True, derive uncertainties on C, P, and F through bootstrap.
        n_bootstraps : int, default=1000
            Number of bootstrap resamples to perform. It defines
            the number of resampling in error estimation.

        Attributes
        ----------
        dict : dict
            Container for storing results based on linear color cuts.
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
        attained_threshold : np.array
            Single-cut threshold attained in the optimization process.
            Each element corresponds to a limiting redshift value.
        """
        self.dataset = dataset.dataset
        self.bands = dataset.bands
        self.colors = dataset.colors
        self.CDZs = dataset.CDZs
        self.PDZ_x = dataset.PDZ_x
        self.weights = np.array(self.dataset["weights"].values)
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

        if isinstance(maximized_quantity_min, (float, int)):
            if not (0.0 <= maximized_quantity_min <= 1.0):
                raise ValueError("maximized_quantity_min must range from 0 to 1.")
            self.maximized_quantity_min = maximized_quantity_min
        else:
            raise TypeError("maximized_quantity_min must be a float or integer.")

        for t in self.thresholded_quantity_min:
            if threshold_buffer > t:
                raise ValueError(
                    "threshold_buffer must be strictly smaller than thresholded_quantity_min."
                )
        self.buffered_threshold = self.thresholded_quantity_min - threshold_buffer

        if np.any(s_samples <= 0):
            raise ValueError(
                "All slope values in s_samples must be strictly positive. "
                f"Found non-positive values: {s_samples[s_samples <= 0]}"
            )
        self.s_samples = s_samples.astype(np.float32)
        self.c_precision = float(c_precision)

        self.k = k
        self.step = step
        self.do_bootstrap = do_bootstrap
        self.n_bootstraps = n_bootstraps

        # Initialize other attributes
        self.dict = {}
        self.dict_steps = {}
        self.P = np.zeros(len(self.td.z_lim))
        self.C = np.zeros(len(self.td.z_lim))
        self.F = np.zeros(len(self.td.z_lim))
        self.C_std = np.full(len(self.td.z_lim), np.nan)
        self.P_std = np.full(len(self.td.z_lim), np.nan)
        self.F_std = np.full(len(self.td.z_lim), np.nan)
        self.attained_threshold = np.zeros(len(self.td.z_lim))

        # Create the target mask
        self.target_mask = self.td._get_target_mask(
            self.dataset, self.td.z_lim, self.CDZs, self.PDZ_x
        )

    def _process_for_one_z(self, args):
        r"""
        Calibrate the linear color cut selection for a single redshift threshold `z_l`.

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

        Returns
        -------
        - float
             redshift threshold
        - dict
             A dictionary containing the optimal cuts and their associated
             performance metrics (purity, completeness, failure rate)
             for the given redshift threshold.
        """
        i_z, store_iterations, cpu_number = args
        z_l = self.td.z_lim[i_z]

        # Minimum required total purity or completeness for the optimized set of cuts
        threshold_tot_min = self.thresholded_quantity_min[i_z]

        # Pre-compute color cut performance
        measure = MeasureLinear(
            self.dataset,
            self.colors,
            self.td,
            self.thresholded_quantity,
            self.maximized_quantity_min,
            self.buffered_threshold[i_z],
            self.s_samples,
            self.c_precision,
            self.step,
        )
        measure.process(z_l, self.CDZs, self.PDZ_x, cpu_number=cpu_number)

        # Get the correct dictionary
        self.dict[z_l] = {}
        all_cuts = measure.dict_all.get(z_l, {})
        if not all_cuts:
            return z_l, None

        # Find the initial threshold
        start_threshold = 0.0
        for base_pair, comp_dict in all_cuts.items():
            for comp_pair, conditions in comp_dict.items():
                for cond_index, cond_data in conditions.items():
                    candidate_max = cond_data.get(
                        self.thresholded_quantity,
                        [],
                    )
                    for i, candidate in enumerate(candidate_max):
                        if candidate > start_threshold:
                            start_threshold = candidate

        if start_threshold <= threshold_tot_min:
            raise ValueError(
                f"At z_l={z_l}, the minimum value of {self.thresholded_quantity}"
                f" attained by single cuts, {start_threshold},"
                f" is <= than the required global threshold, {threshold_tot_min}."
            )

        # Start the iterative process
        if self.thresholded_quantity == "purity":
            maximized_quantity = "completeness"
        else:
            maximized_quantity = "purity"

        current_thresh = 1.0
        best_by_key = {}
        dict_steps = {}

        C_tot = np.zeros_like(self.td.z_lim, dtype=np.float32)
        P_tot = np.zeros_like(self.td.z_lim, dtype=np.float32)
        F_tot = np.zeros_like(self.td.z_lim, dtype=np.float32)

        iteration = 0
        while current_thresh > 0:

            if current_thresh <= self.buffered_threshold[i_z]:
                errMsg = f"For z_l={z_l}, the current threshold in the iterative process is lower"
                errMsg += " than or equal to the buffered threshold."
                errMsg += f" The current global P, C, and F are {P_tot[i_z]}, {C_tot[i_z]}, and {F_tot[i_z]}, respectively."
                errMsg += " Try to increase threshold_buffer in the constructor."
                errMsg += " Consider also the possibility that your requirements can not be met."
                errMsg += " For example, the minimum global purity (or completeness) set by the user could be too low."
                errMsg += " This can happen for low-redshift background selections, where the number"
                errMsg += " of contaminants is low."
                errMsg += " To check this, try to set threshold_buffer equal to thresholded_quantity_min."
                raise ValueError(errMsg)

            if current_thresh > start_threshold:
                current_thresh -= self.step
                continue  # apparently sub-optimal, but essential for consistency with MeasureLinear

            new_best_by_key = {}

            for base_pair, comp_cuts in all_cuts.items():
                for comp_pair, metrics in comp_cuts.items():

                    # simple cuts
                    if comp_pair == base_pair:
                        for cond_index in (0, 1):

                            if cond_index not in metrics:
                                continue

                            cond_dict = metrics[cond_index]

                            c_vals = cond_dict.get("c", [])
                            P_vals = cond_dict.get("purity", [])
                            C_vals = cond_dict.get("completeness", [])

                            for i, c_val in enumerate(c_vals):

                                purity = P_vals[i]
                                compl = C_vals[i]

                                key = (base_pair, comp_pair, cond_index)

                                if self.thresholded_quantity == "purity":
                                    threshold_condition = purity >= current_thresh
                                    maximization_condition = (
                                        compl
                                        > new_best_by_key.get(key, {"completeness": 1})[
                                            "completeness"
                                        ]
                                    )
                                    maximized_quantity_value = compl
                                else:
                                    threshold_condition = compl >= current_thresh
                                    maximization_condition = (
                                        purity
                                        > new_best_by_key.get(key, {"purity": 1})[
                                            "purity"
                                        ]
                                    )
                                    maximized_quantity_value = purity

                                if threshold_condition:
                                    cut = {
                                        "base_pair": base_pair,
                                        "comp_pair": comp_pair,
                                        "c": float(c_val),
                                        "condition_index": cond_index,
                                        maximized_quantity: maximized_quantity_value,
                                    }
                                    if (
                                        key not in new_best_by_key
                                        or maximization_condition
                                    ):
                                        new_best_by_key[key] = cut

                    # cross cuts
                    else:

                        for cond_index, cond_dict in metrics.items():
                            s_vals = cond_dict.get("s", [])
                            c_vals = cond_dict.get("c", [])
                            P_vals = cond_dict.get("purity", [])
                            C_vals = cond_dict.get("completeness", [])

                            for i in range(len(c_vals)):

                                purity = P_vals[i]
                                compl = C_vals[i]

                                key = (base_pair, comp_pair, cond_index)

                                if self.thresholded_quantity == "purity":
                                    threshold_condition = purity >= current_thresh
                                    maximization_condition = (
                                        compl
                                        > new_best_by_key.get(key, {"completeness": 1})[
                                            "completeness"
                                        ]
                                    )
                                    maximized_quantity_value = compl
                                else:
                                    threshold_condition = compl >= current_thresh
                                    maximization_condition = (
                                        purity
                                        > new_best_by_key.get(key, {"purity": 1})[
                                            "purity"
                                        ]
                                    )
                                    maximized_quantity_value = purity

                                if threshold_condition:
                                    cut = {
                                        "base_pair": base_pair,
                                        "comp_pair": comp_pair,
                                        "s": float(s_vals[i]),
                                        "c": float(c_vals[i]),
                                        "condition_index": cond_index,
                                        maximized_quantity: maximized_quantity_value,
                                    }
                                    if (
                                        key not in new_best_by_key
                                        or maximization_condition
                                    ):
                                        new_best_by_key[key] = cut

            if new_best_by_key == {}:
                current_thresh -= self.step
                continue

            C_tot_new = np.zeros_like(self.td.z_lim, dtype=np.float32)
            P_tot_new = np.zeros_like(self.td.z_lim, dtype=np.float32)
            F_tot_new = np.zeros_like(self.td.z_lim, dtype=np.float32)

            nested = build_nested_dict(new_best_by_key)

            combine_optimal_cuts(
                self.td.z_lim,
                self.color_cache,
                self.target_mask,
                self.weights,
                {z_l: nested},
                C_tot_new,
                P_tot_new,
                F_tot_new,
            )

            if self.thresholded_quantity == "purity":
                threshold_condition = P_tot_new[i_z] >= threshold_tot_min
            else:
                threshold_condition = C_tot_new[i_z] >= threshold_tot_min

            if threshold_condition:
                best_by_key = new_best_by_key
                C_tot = C_tot_new
                P_tot = P_tot_new
                F_tot = F_tot_new
                current_thresh -= self.step

                if store_iterations:
                    dict_steps[iteration] = nested
                    iteration += 1
            else:
                if F_tot_new[i_z] == 1.0:
                    errMsg = f"No effective selection found for z_l={z_l}:"
                    errMsg += " the failure rate is 1."
                    raise warnings.warn(errMsg, UserWarning)
                break  # below the threshold

        if current_thresh <= 0:
            errMsg = f"For z_l={z_l}, the convergence is attained for a value larger than the user-defined thresholded_quantity_min."
            errMsg += f" The current global P, C, and F are {P_tot[i_z]}, {C_tot[i_z]}, and {F_tot[i_z]}, respectively."
            raise ValueError(errMsg)

        self.attained_threshold[i_z] = current_thresh
        return z_l, best_by_key, dict_steps

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

            indices = np.random.choice(
                len(self.dataset), size=len(self.dataset), replace=True
            )
            bt_sample = self.dataset.iloc[indices]

            # Pre-compute colors
            color_cache = {}
            for mag1, mag2 in self.colors:
                color_cache[(mag1, mag2)] = bt_sample[mag1].values.astype(
                    np.float32
                ) - bt_sample[mag2].values.astype(np.float32)

            # Get the target mask
            target_mask = self.td._get_target_mask(
                bt_sample, self.td.z_lim, self.CDZs, self.PDZ_x
            )

            # Compute the metrics
            C_temp = np.zeros(len(self.td.z_lim))
            P_temp = np.zeros(len(self.td.z_lim))
            F_temp = np.zeros(len(self.td.z_lim))

            combine_optimal_cuts(
                self.td.z_lim,
                color_cache,
                target_mask,
                self.weights,
                self.dict,
                C_temp,
                P_temp,
                F_temp,
            )

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
        Perform the iterative search for optimal linear color conditions.

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
        print("\nRunning the linear color cut calibration...")

        # Pre-compute colors
        self.color_cache = {}
        for mag1, mag2 in self.colors:
            self.color_cache[(mag1, mag2)] = self.dataset[mag1].values.astype(
                np.float32
            ) - self.dataset[mag2].values.astype(np.float32)

        # Optimized selection
        results = [
            self._process_for_one_z((i_z, store_iterations, cpu_number))
            for i_z in tqdm(range(len(self.td.z_lim)), desc="Looping over z_lim points")
        ]

        for result in results:
            z_l, cuts_dict, steps = result
            self.dict[z_l] = build_nested_dict(cuts_dict)
            self.dict_steps[z_l] = steps

        combine_optimal_cuts(
            self.td.z_lim,
            self.color_cache,
            self.target_mask,
            self.weights,
            self.dict,
            self.C,
            self.P,
            self.F,
        )

        del self.color_cache

        # Compute bootstrap errors
        if self.do_bootstrap:
            self._estimate_errors()

        # Restructure self.dict and interpolate the color cut parameters
        restructure_dict(
            self.dict,
            self.bands,
            self.colors,
            self.td.z_lim,
            self.thresholded_quantity,
            self.thresholded_quantity_min,
            is_zlim_lower=self.td.is_zlim_lower,
        )
        interpolate_parameters(self.dict, k=self.k)
        print("Linear cut calibration complete.")

    def write(
        self,
        output_dir,
        file_name,
        reference_sample_name="",
        notes="",
        write_metrics=True,
        write_redshift_distributions=True,
        n_bins_redshift=200,
        min_redshift=0.0,
        max_redshift=4.0,
    ):
        """
        Save the optimal color selection.

        The dictionary of optimal cuts is saved to a YAML file.
        Metrics and redshift distributions, are saved to CSV files.

        Parameters
        ----------
        output_dir : str
            Path to the directory where the file will be saved.
        file_name : str
            Base name of the output file (without extension).
            All file names start with "FOMO-Z_".
        reference_sample_name : str, optional
            Name of the reference sample, stored in the
            ``reference_sample`` entry of the dictionary.
        notes : str, optional
            Additional notes entering the metadata.
        write_metrics : bool, optional
            If True, write the `FOMO-Z_{file_name}_metrics.csv` file containing
            purity, completeness, and failure rate
            as a function of limiting redshift, and their errors.
        write_redshift_distributions : bool, optional
            If True, write the `FOMO-Z_{file_name}_redshift_distributions.csv` file,
            containing the binned redshift distributions of color-selected objects.
            This file contains the redshift bins and the histograms for each limiting
            redshift point provided to `TargetDefinition`.
            Histograms are derived using the `get_redshift_distribution` method.
            The first two columns of the file contain bin edges and bin centers,
            respectively. Subsequent columns contain the redshift distributions
            corresponding to each limiting redshift value provided to `TargetDefinition`.
            These histograms are normalized such that their integral is 1. All histograms
            share the same bin edges.
        n_bins_redshift : int, optional
            Number of redshift histogram bins. Used only if `write_redshift_distributions`
            is True.
        min_redshift : float, optional
            Minimum redshift. Used only if `write_redshift_distributions` is True.
        max_redshift : float, optional
            Maximum redshift. Used only if `write_redshift_distributions` is True.

        Raises
        ------
        ValueError
            If calibration has not been run.
        """
        if np.sum(self.C) == 0.0:
            raise ValueError("No calibration is available! Run process() first.")

        dictionary = copy.deepcopy(self.dict)

        # Remove unnecessary keys
        for key in dictionary:
            if key != "metadata":
                dictionary[key].pop("c_optimal", None)
                dictionary[key].pop("s_optimal", None)

        # Make the dictionary yaml-serializable
        dictionary = make_yaml_serializable(dictionary)

        # Additional metadata
        dictionary["metadata"]["reference_sample"] = reference_sample_name
        dictionary["metadata"]["notes"] = "Produced with the FOMO-Z code. " + notes

        # Write the dictionary
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"FOMO-Z_{file_name}.yaml")

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dictionary, f, sort_keys=False, allow_unicode=True)

        # Write metrics
        if write_metrics:

            metrics_data = {
                "z_lim": self.td.z_lim,
                "C": self.C,
                "P": self.P,
                "F": self.F,
                "C_std": self.C_std,
                "P_std": self.P_std,
                "F_std": self.F_std,
            }

            metrics_df = pd.DataFrame(metrics_data)
            file_path = os.path.join(output_dir, f"FOMO-Z_{file_name}_metrics.csv")
            metrics_df.to_csv(file_path, index=False)

        # Write the redshift distributions
        if write_redshift_distributions:

            bin_edges = np.linspace(min_redshift, max_redshift, n_bins_redshift + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            dist_df = pd.DataFrame(
                {
                    "bin_edge_low": bin_edges[:-1],
                    "bin_edge_high": bin_edges[1:],
                    "bin_center": bin_centers,
                }
            )

            for i in range(len(self.td.z_lim)):
                hist = self.get_redshift_distribution(i, bin_edges, normalize=True)
                dist_df[f"z_lim_{i}"] = hist

            file_path = os.path.join(
                output_dir, f"FOMO-Z_{file_name}_redshift_distributions.csv"
            )
            dist_df.to_csv(file_path, index=False)

    def get_keys(self):
        """
        Get the optimal color condition keys.

        Returns the list of optimal color condition keys contained in `self.dict`.

        Returns
        ------
        keys : list of str
            List of optimal color condition keys.

        Raises
        ------
        ValueError
            If calibration has not been run.
        """
        if np.sum(self.C) == 0.0:
            raise ValueError("No calibration is available! Run process() first.")

        return [key for key in list(self.dict.keys()) if key != "metadata"]

    def get_mask(self, zl_index, color_condition="all"):
        """
        Get the calibration mask.

        Returns the calibration mask relative to the input limiting redshift.

        Parameters
        ------
        zl_index : int
            Index of the redshift threshold in TargetDefinition.z_lim.
        color_condition : str, optional
            It can be either "all" (all optimal color conditions) or one of
            the optimal color conditions. The latter can be obtained from `self.get_keys()`.

        Returns
        -------
        mask : numpy.array
            Array of booleans.
        """
        assert isinstance(
            color_condition, str
        ), f"color_condition must be a string, got {type(color_condition).__name__}"

        if color_condition == "all":
            color_condition = self.get_keys()
        else:
            if color_condition not in self.get_keys():
                raise ValueError(
                    f"Not allowed color_condition value {color_condition}."
                )
            color_condition = [color_condition]

        z_l = self.td.z_lim[zl_index]
        mags = {
            band: self.dataset[band].to_numpy()
            for band in self.dict["metadata"]["bands"]
        }
        mask = np.full(len(self.dataset), False)
        local_dict = dict(mags)

        for key in color_condition:
            entry = self.dict[key]
            knots = np.unique(entry["knots"])
            if z_l not in knots:
                continue
            idx = np.where(knots == z_l)[0][0]
            local_dict["c"] = entry["c_optimal"][idx]
            if "s_optimal" in entry:
                local_dict["s"] = entry["s_optimal"][idx]
            mask |= ne.evaluate(key, local_dict=local_dict)

        return mask

    def get_redshift_distribution(self, zl_index, bin_edges, normalize=True):
        """
        Get the redshift distribution of color-selected objects.

        Returns the redshift distribution relative to the input limiting redshift.

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
        mask = self.get_mask(zl_index, color_condition="all")
        dataset = self.dataset["redshift"][mask]
        hist, _ = np.histogram(dataset, bins=bin_edges, density=normalize)
        return hist
