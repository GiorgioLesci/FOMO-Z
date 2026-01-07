"""
Derive optimal color selections.

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

from .tools import (
    Measure,
    compute_metrics,
    combine_optimal_cuts,
    build_nested_dict,
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


class CalibrateLinear:
    """Calibrate linear color selections."""

    def __init__(self, data_model):
        r"""
        Class calibrating optimal linear color selections.

        This class provides structures for ranking, aggregating, and optimizing color
        cuts. The maximized quantity (either completeness or purity) is defined by the
        `DataModel` instance given in input to this constructor.
        The method implemented here incrementally relaxes the single color cut purity (completeness)
        threshold in order to identify a set of cuts that maximize total completeness
        (purity) while still satisfying the global threshold. The algorithm
        proceeds iteratively, starting from a strict initial threshold and reducing
        it by `DataModel.step` until the total purity (or completeness) remains above
        `DataModel.thresholded_quantity_min`.
        The starting threshold corresponds to the maximum
        purity (or completeness) attained across all single color conditions.
        Also bootstrap errors on the metrics are derived.

        Constraints
        -----------
        For each redshift value, this method searches for the set of
        cuts that ($\mathcal{P}_{\rm iter}^{\rm ini}$: initial threshold,
        $\delta\mathcal{P}_{\rm iter}$: `DataModel.step`, $z_{\rm l}$: `z_l`,
        $\mathcal{P}_{\rm min}$: `DataModel.thresholded_quantity_min`):
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

        Single color conditions having failure rate equal to 1 or completeness equal
        to 0 are always excluded.

        Attributes
        ----------
        dm : DataModel
            `DataModel` object.
        dict : dict
            Container for storing results based on linear color cuts.
        P : numpy.ndarray
            Total purity values.
        C : numpy.ndarray
            Total completeness values.
        F : numpy.ndarray
            Total failure rates.
        P_std : numpy.ndarray
            Total purity error.
        C_std : numpy.ndarray
            Total completeness error.
        F_std : numpy.ndarray
            Total failure rate error.

        Parameters
        ----------
        data_model : DataModel
            A `DataModel` object containing the dataset, band information,
            redshift thresholds, and precomputed cut metrics.
        """
        self.dm = data_model

        self.dict = {}
        self.dict_steps = {}
        self.P = np.zeros(len(self.dm.z_lim))
        self.C = np.zeros(len(self.dm.z_lim))
        self.F = np.zeros(len(self.dm.z_lim))
        self.C_std = np.full(len(self.dm.z_lim), np.nan)
        self.P_std = np.full(len(self.dm.z_lim), np.nan)
        self.F_std = np.full(len(self.dm.z_lim), np.nan)

    def _process_for_one_z(self, args):
        r"""
        Calibrate the linear color cut selection for a single redshift threshold `z_l`.

        See the documentation of `process` for a detailed description of the
        iterative process implemented in this function.

        Parameters
        ----------
        args : tuple
            A 4-element tuple containing:
            - z_l : float
                The redshift threshold being processed.
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
        z_l, store_iterations, cpu_number = args

        # Minimum required total purity or completeness for the optimized set of cuts
        idx = np.where(self.dm.z_lim == z_l)[0][0]
        threshold_tot_min = self.dm.thresholded_quantity_min[idx]

        # Pre-compute color cut performance
        measure = Measure(self.dm)
        measure.process(z_l, cpu_number=cpu_number)

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
                        self.dm.thresholded_quantity,
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
        if self.dm.thresholded_quantity == "purity":
            maximized_quantity = "completeness"
        else:
            maximized_quantity = "purity"

        current_thresh = start_threshold
        best_by_key = {}
        dict_steps = {}

        C_tot = np.zeros_like(self.dm.z_lim, dtype=np.float32)
        P_tot = np.zeros_like(self.dm.z_lim, dtype=np.float32)
        F_tot = np.zeros_like(self.dm.z_lim, dtype=np.float32)

        iteration = 0
        while current_thresh > 0:

            if current_thresh <= self.dm.buffered_threshold:
                errMsg = "The current threshold in the iterative process is lower"
                errMsg += " than or equal to the buffered threshold set in the"
                errMsg += " DataModel class. Increase threshold_buffer in DataModel."
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

                                if self.dm.thresholded_quantity == "purity":
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

                                if self.dm.thresholded_quantity == "purity":
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

            C_tot_new = np.zeros_like(self.dm.z_lim, dtype=np.float32)
            P_tot_new = np.zeros_like(self.dm.z_lim, dtype=np.float32)
            F_tot_new = np.zeros_like(self.dm.z_lim, dtype=np.float32)

            nested = build_nested_dict(new_best_by_key)

            combine_optimal_cuts(
                self.dm.z_lim,
                self.color_cache,
                self.dm.target_mask,
                {z_l: nested},
                C_tot_new,
                P_tot_new,
                F_tot_new,
            )

            if F_tot_new[idx] == 1.0:
                err_msg = "No effective selection found:"
                err_msg += f" the failure rate is 1 at z_l={z_l}."
                raise ValueError(err_msg)

            if self.dm.thresholded_quantity == "purity":
                threshold_condition = P_tot_new[idx] >= threshold_tot_min
            else:
                threshold_condition = C_tot_new[idx] >= threshold_tot_min

            if threshold_condition:
                best_by_key = new_best_by_key
                C_tot = C_tot_new
                P_tot = P_tot_new
                F_tot = F_tot_new
                current_thresh -= self.dm.step

                if store_iterations:
                    dict_steps[iteration] = nested
                    iteration += 1
            else:
                break  # below the threshold

        return z_l, best_by_key, dict_steps

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
        if np.sum(self.C) == 0.0:
            raise ValueError("No calibration is available! Run process() first.")

        C_stats = np.zeros((2, len(self.dm.z_lim)))  # [sum, sum_of_squares]
        P_stats = np.zeros((2, len(self.dm.z_lim)))
        F_stats = np.zeros((2, len(self.dm.z_lim)))

        for i in tqdm(range(n_bootstraps), desc="Deriving bootstrap errors"):

            indices = np.random.choice(
                len(self.dm.dataset), size=len(self.dm.dataset), replace=True
            )
            bootstrap_sample = self.dm.dataset.iloc[indices]

            # Pre-compute colors
            color_cache = {}
            for mag1, mag2 in self.dm.colors:
                color_cache[(mag1, mag2)] = bootstrap_sample[mag1].values.astype(
                    np.float32
                ) - bootstrap_sample[mag2].values.astype(np.float32)

            C_temp = np.zeros(len(self.dm.z_lim))
            P_temp = np.zeros(len(self.dm.z_lim))
            F_temp = np.zeros(len(self.dm.z_lim))

            # Get the target mask
            target_mask = self.dm.get_target_mask(bootstrap_sample, self.dm.z_lim)

            # Compute the metrics
            combine_optimal_cuts(
                self.dm.z_lim,
                color_cache,
                target_mask,
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

            del bootstrap_sample, C_temp, P_temp, F_temp

        C_mean = C_stats[0] / n_bootstraps
        P_mean = P_stats[0] / n_bootstraps
        F_mean = F_stats[0] / n_bootstraps
        self.C_std = np.sqrt((C_stats[1] / n_bootstraps) - C_mean**2)
        self.P_std = np.sqrt((P_stats[1] / n_bootstraps) - P_mean**2)
        self.F_std = np.sqrt((F_stats[1] / n_bootstraps) - F_mean**2)

    def process(
        self,
        cpu_number=None,
        store_iterations=False,
    ):
        r"""
        Perform the iterative search for optimal linear color conditions.

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
        print("\nRunning the linear color cut calibration...")

        # Pre-compute colors
        self.redshift = self.dm.dataset["redshift"].values.astype(np.float32)
        self.color_cache = {}
        for mag1, mag2 in self.dm.colors:
            self.color_cache[(mag1, mag2)] = self.dm.dataset[mag1].values.astype(
                np.float32
            ) - self.dm.dataset[mag2].values.astype(np.float32)

        # Optimized selection
        results = [
            self._process_for_one_z((z_l, store_iterations, cpu_number))
            for z_l in tqdm(self.dm.z_lim, desc="Looping over z_lim points")
        ]

        for result in results:
            z_l, cuts_dict, steps = result
            self.dict[z_l] = build_nested_dict(cuts_dict)
            self.dict_steps[z_l] = steps

        combine_optimal_cuts(
            self.dm.z_lim,
            self.color_cache,
            self.dm.target_mask,
            self.dict,
            self.C,
            self.P,
            self.F,
        )

        del self.redshift, self.color_cache

        # Bootstrap errors
        if self.dm.do_bootstrap:
            self._estimate_errors(n_bootstraps=self.dm.n_bootstraps)

        #
        restructure_dict(
            self.dict,
            self.dm.bands,
            self.dm.colors,
            self.dm.z_lim,
            self.dm.thresholded_quantity,
            self.dm.thresholded_quantity_min,
            is_zlim_lower=self.dm.is_zlim_lower,
        )
        interpolate_parameters(self.dict, k=self.dm.k)
        print("Linear cut calibration complete.")

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
            All file names start with "FOMO-Z_".
        reference_sample_name : str, optional
            Name of the reference sample, stored in the
            ``reference_sample`` entry of the dictionary.
        notes : str, optional
            Additional notes entering the metadata.
        write_also_metrics : bool, optional
            If True, write the `FOMO-Z_{file_name}_metrics.csv` file containing
            purity, completeness, and failure rate
            as a function of limiting redshift, and their errors.

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

        # Write
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"FOMO-Z_{file_name}.yaml")

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dictionary, f, sort_keys=False, allow_unicode=True)

        # Write metrics
        if write_also_metrics:

            metrics_data = {
                "z_lim": self.dm.z_lim,
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
