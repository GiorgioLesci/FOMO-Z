"""
OPTIMIZE COLOR CUT COMBINATION.

This module selects the optimal color conditions and the corresponding
optimal parameters, given any limit on redshift and on other galaxy properties.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import copy
import multiprocessing as mp
import os
import time
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm
import pandas as pd

from .auxiliary import (
    get_target_mask,
    compute_metrics,
    build_nested_dict,
    float_representer,
    interpolate_parameters,
    make_yaml_serializable,
    restructure_dict,
    set_plot,
)

# Register the custom float representer for both default and safe dumpers
yaml.add_representer(float, float_representer)
yaml.add_representer(float, float_representer, Dumper=yaml.SafeDumper)


class Optimize:
    """Optimize color selections."""

    def __init__(self, measure):
        """
        Class for optimizing color cuts based on metrics computed by `Measure`.

        This class takes the results from a `Measure` instance (which stores
        metrics for a grid of color cuts) and provides structures for ranking,
        aggregating, and optimizing those cuts. Single color conditions having
        failure rate equal to 1 or completeness equal to 0 are always excluded.
        The maximized quantity (either completeness or purity) is defined by the
        `Measure` instance given in input to this constructor.

        Attributes
        ----------
        bands : list of str
            Photometric bands used in the analysis, copied from the input
            `Measure` object.
        colors : list of tuple
            List of 2-band combinations used to form colors, copied from
            the `Measure` object.
        dataset : pandas.DataFrame
            The photometric dataset containing magnitude and redshift values.
        z_lim : numpy.ndarray
            Array of redshift thresholds.
        dict_all : dict
            Dictionary of metrics for all tested color–color cuts, inherited from
            the `Measure` object.
        dict_iterative : dict
            Container for storing results based on an iterative threshold.
        P_iterative_tot : numpy.ndarray
            Total purity values (iterative threshold).
        C_iterative_tot : numpy.ndarray
            Total completeness values (iterative threshold).
        F_iterative_tot : numpy.ndarray
            Total failure rates (iterative threshold).

        Parameters
        ----------
        measure : Measure
            A `Measure` object containing the dataset, band information,
            redshift thresholds, and precomputed cut metrics.
        """
        self.bands = measure.bands
        self.colors = measure.colors
        self.dataset = measure.dataset
        self.z_lim = measure.z_lim
        self.dict_all = measure.dict_all
        self.initial_threshold = measure.initial_threshold
        self.thresholded_quantity = measure.thresholded_quantity
        self.target_mask = measure.target_mask
        self.is_zlim_lower = measure.is_zlim_lower
        self.additional_rules = measure.additional_rules

        self.dict_iterative = {}
        self.dict_iterative_steps = {}
        self.P_iterative_tot = np.zeros(len(self.z_lim))
        self.C_iterative_tot = np.zeros(len(self.z_lim))
        self.F_iterative_tot = np.zeros(len(self.z_lim))
        self.C_std = np.full(len(self.z_lim), np.nan)
        self.P_std = np.full(len(self.z_lim), np.nan)
        self.F_std = np.full(len(self.z_lim), np.nan)

    def _process_iterative_for_one_z(self, args):
        r"""
        Optimize the color cut selection for a single redshift threshold `z_l`.

        See the documentation of `process` for a detailed description of the
        iterative process implemented in this function.

        Parameters
        ----------
        args : tuple
            A 4-element tuple containing:
            - z_l : float
                The redshift threshold being processed.
            - threshold_tot_min : float
                Minimum required total purity or completeness for the optimized set of cuts.
            - step : float
                Increment step applied when scanning over purity or completeness thresholds.
            - store_iterations : bool
                If True, store the iteration steps in self.dict_iterative_steps.

        Returns
        -------
        - float
             redshift threshold
        - dict
             A dictionary containing the optimal cuts and their associated
             performance metrics (purity, completeness, failure rate)
             for the given redshift threshold.
        """
        z_l, threshold_tot_min, step, store_iterations = args

        self.dict_iterative[z_l] = {}
        all_cuts = self.dict_all.get(z_l, {})
        if not all_cuts:
            return z_l, None

        idx = np.where(self.z_lim == z_l)[0][0]

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
                f"At z_l={z_l}, the starting threshold, {start_threshold},"
                f" is <= than the global threshold, {threshold_tot_min}."
            )

        # Start the iterative process
        if self.thresholded_quantity == "purity":
            maximized_quantity = "completeness"
        else:
            maximized_quantity = "purity"

        current_thresh = start_threshold
        best_by_key = {}
        dict_iterative_steps = {}

        C_tot = np.zeros_like(self.z_lim, dtype=np.float32)
        P_tot = np.zeros_like(self.z_lim, dtype=np.float32)
        F_tot = np.zeros_like(self.z_lim, dtype=np.float32)

        iteration = 0
        while current_thresh > 0:

            if current_thresh <= self.initial_threshold:
                errMsg = "The current threshold in the iterative process"
                errMsg += " is lower than or equal to the initial"
                errMsg += " threshold set in the Measure class! Lower the latter."
                raise ValueError(errMsg)

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

            C_tot_new = np.zeros_like(self.z_lim, dtype=np.float32)
            P_tot_new = np.zeros_like(self.z_lim, dtype=np.float32)
            F_tot_new = np.zeros_like(self.z_lim, dtype=np.float32)

            nested = build_nested_dict(new_best_by_key)

            self._combine_optimal_cuts(
                self.color_cache,
                self.target_mask,
                {z_l: nested},
                C_tot_new,
                P_tot_new,
                F_tot_new,
            )

            if F_tot_new[idx] == 1.0:
                err_msg = "No effective selection found:"
                err_msg += f" the failure rate is 1 at z_l={z_l}."
                raise ValueError(err_msg)

            if self.thresholded_quantity == "purity":
                threshold_condition = P_tot_new[idx] >= threshold_tot_min
            else:
                threshold_condition = C_tot_new[idx] >= threshold_tot_min

            if threshold_condition:
                best_by_key = new_best_by_key
                C_tot = C_tot_new
                P_tot = P_tot_new
                F_tot = F_tot_new
                current_thresh -= step

                if store_iterations:
                    dict_iterative_steps[iteration] = nested
                    iteration += 1
            else:
                break  # below the threshold

        return z_l, best_by_key, dict_iterative_steps

    def _combine_optimal_cuts(
        self, color_cache, target_mask, dictionary, C_tot, P_tot, F_tot
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
        dictionary : dict
            Dictionary of selected optimal cuts (e.g. `self.dict_iterative`),
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

        for i, z_l in enumerate(self.z_lim):

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

            TP, FP, FN, TN = compute_metrics(mask_total, target_mask[i])

            C_tot[i] = TP / (TP + FN) if TP + FN > 0 else 0.0
            P_tot[i] = TP / (TP + FP) if TP + FP > 0 else 0.0
            F_tot[i] = FP / (FP + TN) if FP + TN > 0 else 0.0

    def _estimate_errors(self, n_bootstraps=1000):
        """
        Estimate bootstrap errors.

        This method computes mean and standard deviation of completeness, purity,
        and failure rate, based on bootstrap resampling. Results are
        stored as instance attributes:
        - C_std: Standard deviation of completeness
        - P_std: Standard deviation of purity
        - F_std: Standard deviation of failure rate

        Parameters
        ----------
        n_bootstraps : int, optional
            Number of bootstrap resamples to perform, by default 1000.

        Raises
        ------
        ValueError
            If calibration has not been run prior to error estimation.
        """
        if np.sum(self.C_iterative_tot) == 0.0:
            raise ValueError("No calibration is available! Run process() first.")

        C_stats = np.zeros((2, len(self.z_lim)))  # [sum, sum_of_squares]
        P_stats = np.zeros((2, len(self.z_lim)))
        F_stats = np.zeros((2, len(self.z_lim)))

        print("\nDeriving bootstrap errors...")
        for i in tqdm(range(n_bootstraps)):

            indices = np.random.choice(
                len(self.dataset), size=len(self.dataset), replace=True
            )
            bootstrap_sample = self.dataset.iloc[indices]

            # Pre-compute colors
            color_cache = {}
            for mag1, mag2 in self.colors:
                color_cache[(mag1, mag2)] = bootstrap_sample[mag1].values.astype(
                    np.float32
                ) - bootstrap_sample[mag2].values.astype(np.float32)

            C_temp = np.zeros(len(self.z_lim))
            P_temp = np.zeros(len(self.z_lim))
            F_temp = np.zeros(len(self.z_lim))

            # Get the target mask
            target_mask = get_target_mask(
                bootstrap_sample,
                self.z_lim,
                self.is_zlim_lower,
                self.additional_rules,
            )

            # Compute the metrics
            self._combine_optimal_cuts(
                color_cache,
                target_mask,
                self.dict_iterative,
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

            del bootstrap_sample, C_temp, P_temp, F_temp

        C_mean = C_stats[0] / n_bootstraps
        P_mean = P_stats[0] / n_bootstraps
        F_mean = F_stats[0] / n_bootstraps
        self.C_std = np.sqrt((C_stats[1] / n_bootstraps) - C_mean**2)
        self.P_std = np.sqrt((P_stats[1] / n_bootstraps) - P_mean**2)
        self.F_std = np.sqrt((F_stats[1] / n_bootstraps) - F_mean**2)

    def process(
        self,
        threshold_tot_min,
        step=0.001,
        k=1,
        do_bootstrap=False,
        n_bootstraps=1000,
        cpu_number=None,
        store_iterations=False,
    ):
        r"""
        Perform the iterative search for optimal color conditions.

        Here, either completeness or purity are maximized, depending on what was
        set in the `Measure` instance given in input to the constructor of this class.
        This method incrementally relaxes the single color cut purity (completeness)
        threshold in order to identify a set of cuts that maximize total completeness
        (purity) while still satisfying the global threshold. The algorithm
        proceeds iteratively, starting from a strict initial threshold and reducing
        it by `step` until the total purity (or completeness) remains above
        `threshold_tot_min`. The starting threshold corresponds to the maximum
        purity (or completeness) attained across all single color conditions.
        Also bootstrap errors on the metrics are derived.

        Constraints
        -----------
        For each redshift value, this method searches for the set of
        cuts that ($\mathcal{P}_{\rm iter}^{\rm ini}$: initial threshold,
        $\delta\mathcal{P}_{\rm iter}$: `step`, $z_{\rm l}$: `z_l`,
        $\mathcal{P}_{\rm min}$: `threshold_tot_min`):
        - Update $\mathcal{P}_{\rm iter}^k = \mathcal{P}_{\rm iter}^{\rm ini} -
          k\,\delta\mathcal{P}_{\rm iter}$, where $k$ is the iteration index,
          starting from 0.
        - For any $(x,y)$ color pair, select the $j$th condition
          if there exists at least one parameter set $\vec{p}$ that yields
          $\mathcal{P}_j(z_{\rm l}\,|\,\vec{p},x,y)>\mathcal{P}_{\rm iter}^k$.
        - Among those selected at step 2, define the optimal $j$th condition
          in the $(x,y)$ space as the one having $\vec{p}$ parameters that
          maximize $\mathcal{C}_j(z_{\rm l}\,|\,\vec{p},x,y)$.
        - Combine the selected conditions in all $(x,y)$ spaces through the
          $\lor$ logical operator, and evaluate the global purity,
          $\mathcal{P}(z_{\rm l})$.
        - Repeat from point 1 as long as
          $\mathcal{P}(z_{\rm l})>\mathcal{P}_{\rm min}$.

        After the optimal cuts are found for all redshift bins, the results are:
        1. Combined into global purity, completeness, and failure rate metrics.
        2. Restructured into an expression-based dictionary.
        3. Interpolated across redshift bins to provide continuous cut parameters.

        Parameters
        ----------
        threshold_tot_min : float or array-like
            Required minimum global purity or completeness across all redshift bins.
            If float, the same threshold is applied at any limiting redshift point.
            If array-like, a different threshold is used at each limiting
            redshift.
        step : float, optional
            Increment by which the single-cut threshold is reduced
            during the search. Default is 0.001.
        cpu_number : int or None, optional
            Number of CPU cores to use for parallel processing. If None,
            defaults to using all available cores minus two.
        k : int, optional
            Requested degree of the spline interpolation. Default is 1 (linear).
            The actual degree is adjusted case by case:
            - If there are fewer than ``k+1`` points, the degree is reduced
              to ``n_points - 1``.
            - If there is only one point, interpolation is skipped.
        do_bootstrap : bool, optional
            If True, derive uncertainties on C, P, and F through bootstrap.
        n_bootstraps : int, optional
            Number of bootstrap resamples to perform, by default 1000.
        store_iterations : bool, optional
            If True, store the iteration steps in self.dict_iterative_steps.
            Default is False.

        Returns
        -------
        None
            Updates the following instance attributes in place:
            - ``self.dict_iterative`` : dict
                Nested dictionary of optimal iterative cuts for each
                redshift bin, restructured and interpolated.
            - ``self.P_iterative_tot`` : numpy.ndarray
                Total purity across all bins.
            - ``self.C_iterative_tot`` : numpy.ndarray
                Total completeness across all bins.
            - ``self.F_iterative_tot`` : numpy.ndarray
                Total failure rate across all bins.
        """
        if isinstance(threshold_tot_min, float):
            threshold_tot_min = [threshold_tot_min] * len(self.z_lim)

        print("\nRunning iterative threshold search...")
        start_time = time.time()

        # Pre-compute colors
        self.redshift = self.dataset["redshift"].values.astype(np.float32)
        self.color_cache = {}
        for mag1, mag2 in self.colors:
            self.color_cache[(mag1, mag2)] = self.dataset[mag1].values.astype(
                np.float32
            ) - self.dataset[mag2].values.astype(np.float32)

        # Parallel processing
        N_cpu = (
            min(cpu_number, mp.cpu_count() - 2)
            if cpu_number
            else max(mp.cpu_count() - 2, 1)
        )
        N_cpu = max(1, N_cpu)  # Ensure at least 1 process

        args_list = [
            (
                z_l,
                threshold_tot_min[np.where(self.z_lim == z_l)[0][0]],
                step,
                store_iterations,
            )
            for z_l in self.z_lim
        ]
        with mp.Pool(processes=N_cpu) as pool:
            results = pool.map(self._process_iterative_for_one_z, args_list)

        for result in results:
            z_l, cuts_dict, steps = result
            self.dict_iterative[z_l] = build_nested_dict(cuts_dict)
            self.dict_iterative_steps[z_l] = steps

        print("Combining the optimal cuts...")
        self._combine_optimal_cuts(
            self.color_cache,
            self.target_mask,
            self.dict_iterative,
            self.C_iterative_tot,
            self.P_iterative_tot,
            self.F_iterative_tot,
        )

        total_time = time.time() - start_time
        print(
            "Iterative thresholding complete.",
            f"Total time: {total_time:.2f} seconds.",
        )

        del self.redshift, self.color_cache

        # Bootstrap errors
        if do_bootstrap:
            self._estimate_errors(n_bootstraps=n_bootstraps)

        #
        restructure_dict(
            self.dict_iterative,
            self.bands,
            self.colors,
            self.z_lim,
            self.thresholded_quantity,
            threshold_tot_min,
            is_zlim_lower=self.is_zlim_lower,
        )
        interpolate_parameters(self.dict_iterative, k=k)

    def write_optimal_selection(
        self,
        output_dir,
        file_name,
        reference_sample_name="",
        notes="",
        write_also_metrics=True,
    ):
        """
        Save the optimal color selection to a YAML file.

        The dictionary of iterative-threshold cuts is serialized
        and written to file.

        Parameters
        ----------
        output_dir : str
            Path to the directory where the file will be saved.
        file_name : str
            Base name of the output file (without extension).
        reference_sample_name : str, optional
            Name of the reference sample, stored in the
            ``reference_sample`` entry of the dictionary.
        notes : str, optional
            Additional notes entering the metadata.
        write_also_metrics : bool, optional
            If True, write the `{file_name}_metrics.csv` file containing
            purity, completeness, and failure rate
            as a function of limiting redshift, and their errors.

        Raises
        ------
        ValueError
            If calibration has not been run.
        """
        if np.sum(self.C_iterative_tot) == 0.0:
            raise ValueError("No calibration is available! Run process() first.")

        dictionary = self.dict_iterative

        # Make the dictionary yaml-serializable
        dictionary = make_yaml_serializable(dictionary)

        # Additional metadata
        dictionary["metadata"]["reference_sample"] = reference_sample_name
        dictionary["metadata"]["notes"] = (
            "Produced with the FOMO-Z Python code. " + notes
        )

        # Write
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{file_name}.yaml")

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dictionary, f, sort_keys=False, allow_unicode=True)

        # Write metrics
        if write_also_metrics:

            metrics_data = {
                "z_lim": self.z_lim,
                "C": self.C_iterative_tot,
                "P": self.P_iterative_tot,
                "F": self.F_iterative_tot,
                "C_std": self.C_std,
                "P_std": self.P_std,
                "F_std": self.F_std,
            }

            metrics_df = pd.DataFrame(metrics_data)
            file_path = os.path.join(output_dir, f"{file_name}_metrics.csv")
            metrics_df.to_csv(file_path, index=False)

    def plot(self, loclegend="upper right"):
        """
        Plot the performance metrics for the optimized color cuts.

        This method produces a three-panel figure showing:
        1. Completeness as a function of redshift threshold.
        2. Purity as a function of redshift threshold.
        3. Failure rate as a function of redshift threshold.

        Parameters
        ----------
        loclegend : str, optional
            Location of the legend in the first panel. Default is "upper right".

        Returns
        -------
        None
            Displays the matplotlib figure.
        """
        ax = set_plot()

        if np.sum(self.C_iterative_tot) != 0.0:
            ax[0].plot(
                self.z_lim,
                self.C_iterative_tot * 100.0,
                color="k",
                linestyle="-",
                label="Optimized selection",
            )
            ax[1].plot(
                self.z_lim,
                self.P_iterative_tot * 100.0,
                color="k",
                linestyle="-",
            )
            ax[2].plot(
                self.z_lim,
                self.F_iterative_tot * 100.0,
                color="k",
                linestyle="-",
            )

        # Add error bands if available
        if not np.any(np.isnan(self.C_std)):
            ax[0].fill_between(
                self.z_lim,
                (self.C_iterative_tot - self.C_std) * 100.0,
                (self.C_iterative_tot + self.C_std) * 100.0,
                color="k",
                alpha=0.3,
                label="Bootstrap error",
            )
            ax[1].fill_between(
                self.z_lim,
                (self.P_iterative_tot - self.P_std) * 100.0,
                (self.P_iterative_tot + self.P_std) * 100.0,
                color="k",
                alpha=0.3,
            )
            ax[2].fill_between(
                self.z_lim,
                (self.F_iterative_tot - self.F_std) * 100.0,
                (self.F_iterative_tot + self.F_std) * 100.0,
                color="k",
                alpha=0.3,
            )

        ax[0].legend(loc=loclegend, fontsize=13)
        plt.show()
