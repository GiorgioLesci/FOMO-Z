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


class CalibrateLinear:
    """Calibrate linear color selections."""

    def __init__(
        self,
        dataset: Dataset,
        target_definition: TargetDefinition,
        thresholded_quantity_min,
        maximized_quantity="completeness",
        maximized_quantity_min=0.05,
        threshold_buffer=0.05,
    ):
        r"""
        Class calibrating optimal linear color selections.

        This class provides structures for ranking, aggregating, and optimizing color
        cuts. The maximized quantity (either completeness or purity) is defined by the
        `TargetDefinition` instance given in input to this constructor.
        The method implemented here incrementally relaxes the single color cut purity (completeness)
        threshold in order to identify a set of cuts that maximize total completeness
        (purity) while still satisfying the global threshold. The algorithm
        proceeds iteratively, starting from a strict initial threshold and reducing
        it by `TargetDefinition.step` until the total purity (or completeness) remains above
        `thresholded_quantity_min`.
        The starting threshold corresponds to the maximum
        purity (or completeness) attained across all single color conditions.
        Also bootstrap errors on the metrics are derived.

        Constraints
        -----------
        For each redshift value, this method searches for the set of
        cuts that ($\mathcal{P}_{\rm iter}^{\rm ini}$: initial threshold,
        $\delta\mathcal{P}_{\rm iter}$: `TargetDefinition.step`, $z_{\rm l}$: `z_l`,
        $\mathcal{P}_{\rm min}$: `thresholded_quantity_min`):
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
            and reduce statistical uncertainties on selection purity and completeness.
        threshold_buffer : float, default=0.05
            Quantity subtracted from `thresholded_quantity_min`.
            Color conditions not satisfying the final threshold, namely
            `thresholded_quantity_min - threshold_buffer`, are not
            considered in the optimization process. This is useful to avoid filling memory
            with unused information. The default value is 0.05, namely a 5 percentage point
            subtraction. If this value is too small, the optimization method in
            `CalibrateLinear` raises an error.

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
        self.td = target_definition

        if isinstance(thresholded_quantity_min, float):
            self.thresholded_quantity_min = [thresholded_quantity_min] * len(
                self.td.z_lim
            )
        else:
            if len(thresholded_quantity_min) != len(self.td.z_lim):
                raise ValueError(
                    f"thresholded_quantity_min length ({len(thresholded_quantity_min)}) "
                    f"must match z_lim length ({len(self.td.z_lim)})"
                )
            self.thresholded_quantity_min = thresholded_quantity_min
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

        if threshold_buffer > thresholded_quantity_min:
            raise ValueError(
                "threshold_buffer must be strictly smaller than thresholded_quantity_min."
            )
        self.buffered_threshold = thresholded_quantity_min - threshold_buffer

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
        self.target_mask = self.td.get_target_mask(self.dataset, self.td.z_lim)

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
            self.buffered_threshold,
        )
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
        dict_steps = {}

        C_tot = np.zeros_like(self.td.z_lim, dtype=np.float32)
        P_tot = np.zeros_like(self.td.z_lim, dtype=np.float32)
        F_tot = np.zeros_like(self.td.z_lim, dtype=np.float32)

        iteration = 0
        while current_thresh > 0:

            if current_thresh <= self.buffered_threshold:
                errMsg = "The current threshold in the iterative process is lower"
                errMsg += " than or equal to the buffered threshold set in the"
                errMsg += " TargetDefinition class. Increase threshold_buffer in the constructor."
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

            C_tot_new = np.zeros_like(self.td.z_lim, dtype=np.float32)
            P_tot_new = np.zeros_like(self.td.z_lim, dtype=np.float32)
            F_tot_new = np.zeros_like(self.td.z_lim, dtype=np.float32)

            nested = build_nested_dict(new_best_by_key)

            combine_optimal_cuts(
                self.td.z_lim,
                self.color_cache,
                self.target_mask,
                {z_l: nested},
                C_tot_new,
                P_tot_new,
                F_tot_new,
            )

            if F_tot_new[i_z] == 1.0:
                err_msg = "No effective selection found:"
                err_msg += f" the failure rate is 1 at z_l={z_l}."
                raise ValueError(err_msg)

            if self.thresholded_quantity == "purity":
                threshold_condition = P_tot_new[i_z] >= threshold_tot_min
            else:
                threshold_condition = C_tot_new[i_z] >= threshold_tot_min

            if threshold_condition:
                best_by_key = new_best_by_key
                C_tot = C_tot_new
                P_tot = P_tot_new
                F_tot = F_tot_new
                current_thresh -= self.td.step

                if store_iterations:
                    dict_steps[iteration] = nested
                    iteration += 1
            else:
                break  # below the threshold

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

        for i in tqdm(range(self.td.n_bootstraps), desc="Deriving bootstrap errors"):

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
            target_mask = self.td.get_target_mask(bt_sample, self.td.z_lim)

            # Compute the metrics
            C_temp = np.zeros(len(self.td.z_lim))
            P_temp = np.zeros(len(self.td.z_lim))
            F_temp = np.zeros(len(self.td.z_lim))

            combine_optimal_cuts(
                self.td.z_lim,
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
        self.redshift = self.dataset["redshift"].values.astype(np.float32)
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
            self.dict,
            self.C,
            self.P,
            self.F,
        )

        del self.redshift, self.color_cache

        # Compute bootstrap errors
        if self.td.do_bootstrap:
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
        interpolate_parameters(self.dict, k=self.td.k)
        print("Linear cut calibration complete.")

    def write(
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
