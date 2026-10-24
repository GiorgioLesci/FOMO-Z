"""
Calibrate linear color selections and measure their reference performance.

Support prior-weighted and PDZ-based targets, iterative cut optimization,
bootstrap metric scatter, parameter interpolation, and calibration export.

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

from ..data.dataset import Dataset
from ..data.selection_definition import SelectionDefinition
from ..auxiliary import (
    MeasureLinear,
    build_nested_dict,
    combine_optimal_cuts,
    float_representer,
    inline_list_representer,
    interpolate_parameters,
    make_yaml_serializable,
    restructure_dict,
)
from ..auxiliary.distributions import get_distribution as compute_distribution

# Register the custom representers for both default and safe dumpers
yaml.add_representer(float, float_representer)
yaml.add_representer(float, float_representer, Dumper=yaml.SafeDumper)

yaml.add_representer(list, inline_list_representer)
yaml.add_representer(list, inline_list_representer, Dumper=yaml.SafeDumper)


class LinearClassification:
    """
    Calibrate OR combinations of linear color cuts on a reference sample.

    Attributes
    ----------
    dict : dict
        After process, color-cut expressions and metadata, including
        interpolation knots, coefficients, and stored optimal parameters.
        Metrics are stored separately in the attributes below.
    C, P, F : numpy.ndarray
        Global completeness, purity, and false positive rate at each
        calibration threshold, in SelectionDefinition.z_lim order.
    C_std, P_std, F_std : numpy.ndarray
        Bootstrap standard deviations, or NaN if not estimated.
    attained_threshold : numpy.ndarray
        Single-cut purity threshold at termination for each redshift.
        This is not the attained global purity and may describe a
        rejected trial rather than the retained cut combination.
    dict_steps : dict
        Per-threshold accepted iteration states when requested.
    target_mask : numpy.ndarray
        Reference target memberships including prior weights, with shape
        (n_thresholds, n_reference); boolean for unweighted point targets.
    """

    def __init__(
        self,
        dataset: Dataset,
        selection_definition: SelectionDefinition,
        purity_min,
        completeness_min=0.05,
        threshold_buffer=0.05,
        s_samples=np.linspace(0.3, 2.5, 50),
        c_precision=0.05,
        k=1,
        step=0.0001,
        do_bootstrap=True,
        n_bootstraps=1000,
        verbose=True,
    ):
        """
        Configure an iterative calibration of linear color cuts.

        For each pair of colors ``(x, y)``, candidates include ``x > c``,
        ``x < c``, ``y > c``, ``y < c``, and the four inclined cuts
        ``x > s*y + c``, ``x > -s*y + c``, ``x < s*y + c``, and
        ``x < -s*y + c``, with positive slope magnitudes ``s``. Single-color
        cuts are evaluated once per color. All retained cuts are combined
        through logical OR, both within and across color--color spaces.
        This approach is suited to targets occupying the outskirts of color
        distributions, such as background galaxies for weak lensing.

        Parameters
        ----------
        dataset : Dataset
            Reference sample providing magnitudes, target properties, optional
            PDZs, and prior weights. The internal DataFrame is referenced,
            not copied. Clean required target properties with
            ``Dataset.drop_missing`` before constructing the calibration,
            and avoid modifying the sample during calibration. Magnitude
            uncertainties are not propagated by this binary method.
        selection_definition : SelectionDefinition
            Target criteria and threshold grid. Required properties are
            validated and target membership is evaluated during construction.
            For interpolation across multiple thresholds, supply distinct
            thresholds in increasing order.
        purity_min : float or one-dimensional array-like
            Required global purity in [0, 1). Supply a Python float to use
            the same value at every threshold, or one value per entry in
            ``selection_definition.z_lim``. The combined selection is accepted
            when its purity is greater than or equal to this value.
        completeness_min : float, default=0.05
            Strict lower bound on each candidate cut's completeness, in
            [0, 1]. This is a single-cut constraint, not a threshold imposed
            directly on the combined selection. Candidates with zero purity
            or completeness, or a failure rate of one, are also discarded.
        threshold_buffer : float, default=0.05
            Candidate-search margin below the requested global purity.
            Only candidates with purity strictly above
            ``purity_min - threshold_buffer`` are retained for optimization.
            Use a non-negative margin no larger than the smallest requested
            purity; equality is accepted by the implementation.
            If relaxation reaches this buffered boundary, process raises
            ValueError. Increasing the margin retains more candidates and
            may allow the search to proceed, at greater memory cost.
        s_samples : numpy.ndarray, default=np.linspace(0.3, 2.5, 50)
            One-dimensional array of strictly positive slope magnitudes.
            Both signs are tested. The default has 50 equally spaced values
            from 0.3 to 2.5. Values are stored as float32. Supply a NumPy
            array rather than a Python list.
        c_precision : float, default=0.05
            Positive nominal spacing of the intercept grid, in magnitude
            units. For a data-driven range of width ``width``, the helper
            uses ``int(width / c_precision) + 1`` evenly spaced samples;
            the actual spacing need not equal this value exactly.
            Smaller values increase computational and memory requirements.
        k : int, default=1
            Requested interpolating-spline degree, from 1 to 5. The default
            is linear interpolation. For each cut, the effective degree is
            ``min(k, n_points - 1)``; interpolation is skipped if the cut
            is retained at only one threshold. Interpolation does not
            impose a purity constraint between calibration thresholds.
        step : float, default=0.0001
            Positive decrement of the iterative single-cut purity threshold,
            starting at unity. Smaller values sample the relaxation more
            finely and increase the number of iterations.
        do_bootstrap : bool, default=True
            Estimate metric uncertainties by resampling reference rows and
            applying the already calibrated cuts to each resample.
        n_bootstraps : int, default=1000
            Positive number of bootstrap realizations when enabled.
        verbose : bool, default=True
            Display calibration progress messages and progress bars.

        Raises
        ------
        ValueError
            If purity thresholds have incompatible length or lie outside
            [0, 1), completeness_min lies outside [0, 1], threshold_buffer
            exceeds a purity threshold, or a slope is non-positive.
        TypeError
            If completeness_min is not a supported numerical scalar.

        Notes
        -----
        The current implementation maximizes completeness with a purity
        constraint. This objective is fixed by this class, not chosen by
        SelectionDefinition. Candidate parameters are sampled on finite
        grids; the procedure is an iterative search, not a guarantee of
        the global optimum over all possible combinations of cuts.

        At each iteration, candidates must meet the current single-cut
        purity threshold (inclusive). For each condition, the admissible
        parameters with greatest completeness are chosen. Their OR union
        is evaluated, and accepted if its global purity meets purity_min
        (inclusive). The threshold is then lowered by step. On the first
        combined selection below the global purity requirement, the last
        accepted selection is retained. Iterations without candidates
        simply lower the threshold. Candidate prefiltering enforces the
        strict completeness and buffered-purity bounds described above.

        With prior weights and/or reference PDZs, metrics are ratios of
        weighted counts. Completeness is ``TP / (TP + FN)``, purity is
        ``TP / (TP + FP)``, and failure rate is ``FP / (FP + TN)``.
        Failure rate measures the fraction of non-targets selected, not
        the contamination fraction ``1 - purity``. Undefined global ratios
        with zero denominator are stored as zero by the metric helper.

        Call process before querying masks, distributions, or writing files.
        Use the exported calibration with Classifier to select objects in
        an independent sample. Use GridClassification for selections with
        more general shapes in color--color space.
        """
        if k != 1:
            raise ValueError("Only k=1 (linear interpolation) is currently supported.")

        self.verbose = verbose
        self.dataset = dataset.dataset
        self.bands = dataset.bands
        self.colors = dataset.colors
        self.CDZs = dataset.CDZs
        self.PDZ_x = dataset.PDZ_x
        self.z_gal_string = dataset.z_gal_string
        self.missing_data = dataset.missing_data
        self.weights = np.array(self.dataset["weights"].values)
        self.td = selection_definition

        if isinstance(purity_min, float):
            self.thresholded_quantity_min = np.asarray(
                [purity_min] * len(self.td.z_lim)
            )
        else:
            if len(purity_min) != len(self.td.z_lim):
                raise ValueError(
                    f"purity_min length ({len(purity_min)}) "
                    f"must match z_lim length ({len(self.td.z_lim)})"
                )
            self.thresholded_quantity_min = np.asarray(purity_min)
        for t in self.thresholded_quantity_min:
            if not (0.0 <= t < 1.0):
                raise ValueError("purity_min accepted values lie in the range [0, 1).")

        maximized_quantity = "completeness"  # hard-coded for now
        if maximized_quantity not in ["completeness", "purity"]:
            errMsg = "maximized_quantity must be either 'completeness' or 'purity'."
            raise ValueError(errMsg)
        if maximized_quantity == "completeness":
            self.thresholded_quantity = "purity"
        else:
            raise NotImplementedError("Purity maximization is not implemented yet.")
            self.thresholded_quantity = "completeness"

        if isinstance(completeness_min, (float, int)):
            if not (0.0 <= completeness_min <= 1.0):
                raise ValueError("completeness_min must range from 0 to 1.")
            self.maximized_quantity_min = completeness_min
        else:
            raise TypeError("completeness_min must be a float or integer.")

        for t in self.thresholded_quantity_min:
            if threshold_buffer > t:
                raise ValueError(
                    "threshold_buffer must be strictly smaller than purity_min."
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
            self.dataset,
            self.td.z_lim,
            self.CDZs,
            self.PDZ_x,
            self.z_gal_string,
            self.missing_data,
        )

    def _process_for_one_z(self, args):
        """
        Calibrate one threshold using the cached reference colors.

        Parameters
        ----------
        args : tuple
            ``(i_z, store_iterations, cpu_number)``: integer index into
            ``self.td.z_lim``, whether to retain accepted iteration states,
            and the worker count passed to MeasureLinear.

        Returns
        -------
        z_l : numpy scalar
            Threshold being calibrated.
        best_by_key : dict
            Last accepted cuts keyed by base color, comparison color, and
            condition index. Entries contain cut parameters and individual
            completeness values.
        dict_steps : dict
            Accepted nested cut dictionaries indexed from zero, or an empty
            dictionary if iteration storage is disabled. The index counts
            accepted states, not all purity-threshold decrements.

        Notes
        -----
        This internal method requires the color cache prepared by process.
        It updates attained_threshold with the stopping single-cut threshold,
        which may belong to the first rejected trial rather than the retained
        selection. Candidate evaluation is delegated to MeasureLinear.

        Raises
        ------
        ValueError
            If the maximum candidate purity does not exceed the global
            requirement, the buffered boundary is reached, or relaxation
            exhausts the allowed threshold range.
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
        measure.process(
            z_l,
            self.CDZs,
            self.PDZ_x,
            self.z_gal_string,
            self.missing_data,
            cpu_number=cpu_number,
        )

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
                errMsg += (
                    " To check this, try to set threshold_buffer equal to purity_min."
                )
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
            errMsg = f"For z_l={z_l}, the convergence is attained for a value larger than the user-defined purity_min."
            errMsg += f" The current global P, C, and F are {P_tot[i_z]}, {C_tot[i_z]}, and {F_tot[i_z]}, respectively."
            raise ValueError(errMsg)

        self.attained_threshold[i_z] = current_thresh
        return z_l, best_by_key, dict_steps

    def _estimate_errors(self):
        """
        Estimate metric scatter by bootstrapping the reference rows.

        Each realization draws as many rows as the reference sample, with
        replacement. Object weights and cumulative PDZs follow the sampled
        indices. Target rules are reevaluated and the fixed optimized cuts
        are applied without repeating calibration.

        Returns
        -------
        None
            Store population standard deviations (ddof=0) across bootstrap
            realizations in C_std, P_std, and F_std. Original-sample metrics
            C, P, and F are retained; bootstrap means are not stored.

        Raises
        ------
        ValueError
            If the sum of calibrated completeness values is zero.

        Notes
        -----
        Called internally before the cut dictionary is restructured for
        interpolation. Uses NumPy's global random state. This estimates
        sampling scatter conditional on the fitted cuts; it does not include
        calibration uncertainty or sample variance between independent fields.
        """
        if np.sum(self.C) == 0.0:
            raise ValueError("No calibration is available! Run process() first.")

        C_stats = np.zeros((2, len(self.td.z_lim)))  # [sum, sum_of_squares]
        P_stats = np.zeros((2, len(self.td.z_lim)))
        F_stats = np.zeros((2, len(self.td.z_lim)))

        for i in tqdm(
            range(self.n_bootstraps),
            desc="Deriving bootstrap errors",
            disable=not self.verbose,
        ):

            indices = np.random.choice(
                len(self.dataset), size=len(self.dataset), replace=True
            )
            bt_sample = self.dataset.iloc[indices].copy()

            # Pre-compute colors
            color_cache = {}
            for mag1, mag2 in self.colors:
                color_cache[(mag1, mag2)] = bt_sample[mag1].values.astype(
                    np.float32
                ) - bt_sample[mag2].values.astype(np.float32)

            # Get the target mask
            bt_CDZs = self.CDZs[indices] if self.CDZs is not None else None
            target_mask = self.td._get_target_mask(
                bt_sample,
                self.td.z_lim,
                bt_CDZs,
                self.PDZ_x,
                self.z_gal_string,
                self.missing_data,
            )

            # Compute the metrics
            C_temp = np.zeros(len(self.td.z_lim))
            P_temp = np.zeros(len(self.td.z_lim))
            F_temp = np.zeros(len(self.td.z_lim))

            combine_optimal_cuts(
                self.td.z_lim,
                color_cache,
                target_mask,
                bt_sample["weights"],
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
        """
        Calibrate linear cuts and prepare their parameter interpolants.

        Evaluate candidate cuts, iteratively maximize completeness under the
        global purity requirement, compute reference-sample metrics, optionally
        estimate bootstrap scatter, and restructure the calibration dictionary
        for export.

        Parameters
        ----------
        cpu_number : int or None, default=None
            Requested number of multiprocessing workers for candidate-cut
            evaluation. None uses ``max(multiprocessing.cpu_count() - 2, 1)``.
            Positive requests are capped at the available core count minus
            two, with at least one worker. Thresholds are processed
            sequentially; bootstrap realizations are also sequential.
        store_iterations : bool, default=False
            Retain accepted optimization states in ``dict_steps[z_lim]``.
            Each state is a nested cut dictionary indexed by the accepted
            iteration number. Skipped or rejected trials are not stored.

        Returns
        -------
        None
            Populate dict, C, P, F, attained_threshold, and dict_steps.
            Populate C_std, P_std, and F_std when bootstrap is enabled;
            otherwise their initial NaN values remain.

        Raises
        ------
        ValueError
            If candidate cuts cannot meet the search requirements or the
            iterative search reaches its buffered boundary.

        Notes
        -----
        Metrics are measured at the supplied calibration thresholds before
        parameter interpolation. Each expression is interpolated only using
        thresholds where it was retained. Bootstrap keeps the calibrated cuts
        fixed. For scripts using multiprocessing, call process under an
        ``if __name__ == "__main__":`` guard where required by the platform.
        """
        if self.verbose:
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
            for i_z in tqdm(
                range(len(self.td.z_lim)),
                desc="Looping over z_lim points",
                disable=not self.verbose,
            )
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
        if self.verbose:
            print("Linear cut calibration complete.")

    def get_distribution(self, zl_index, variable, bin_edges, normalize=True):
        """
        Compute a property distribution of color-selected reference objects.

        Combine reference prior weights with the color-selection mask.
        Redshift distributions always use reference PDZs when available.
        Otherwise, use point estimates. Other properties use point values.

        Parameters
        ----------
        zl_index : int
            Index of the threshold in SelectionDefinition.z_lim.
        variable : str
            Name of the reference property to histogram.
            Use "redshift" or the original Dataset.z_gal_string alias
            for redshift. Other column names are read literally.
        bin_edges : array-like of shape (n_bins + 1,)
            Finite, strictly increasing edges in units of variable.
        normalize : bool, default=True
            Return a density normalized over the bin range if True;
            otherwise return weighted counts.

        Returns
        -------
        numpy.ndarray of shape (n_bins,)
            Weighted counts or normalized probability densities.

        Raises
        ------
        TypeError
            If variable is not a non-empty string.
        ValueError
            If point-estimate redshifts are requested without a designated
            redshift column, or if distribution inputs are invalid.
        KeyError
            If the requested point-value column is absent.

        Notes
        -----
        The original non-None Dataset.z_gal_string is resolved to "redshift".

        Reference PDZs describe redshift only. When available, their use for
        redshift reconstruction is mandatory and no point estimates are
        required. With Dataset.z_gal_string=None, only "redshift" selects
        PDZ mode. No other column is automatically treated as an alias.

        The result describes the color-selected reference sample, weighted
        by its prior weights. It does not transfer a distribution to an
        independent test sample. Normalization is conditional on the
        supplied bin range.

        Only color selection and prior weights enter the histogram: it is
        not restricted again to objects satisfying the target definition.
        In PDZ mode, probability is distributed among bins using linearly
        interpolated cumulative distributions. In point-value mode, bins
        are left-inclusive and right-exclusive, except for the last bin,
        which includes its upper edge. Contributing objects must have valid
        property values; zero-weight objects may have missing values.
        Weight outside the supplied bins is excluded and reported through
        a warning. Normalization requires positive weight within the bins.
        """
        if not isinstance(variable, str) or not variable:
            raise TypeError("variable must be a non-empty string.")

        is_redshift = variable == "redshift" or (
            self.z_gal_string is not None and variable == self.z_gal_string
        )
        pdz_mode = is_redshift and self.CDZs is not None

        if pdz_mode:
            values = None
        else:
            if is_redshift and self.z_gal_string is None:
                raise ValueError(
                    "Redshift reconstruction requires reference PDZs "
                    "or a point-estimate column designated through "
                    "Dataset.z_gal_string."
                )

            key = "redshift" if is_redshift else variable
            if key not in self.dataset.columns:
                raise KeyError(f"Reference property column {key!r} is missing.")
            values = self.dataset[key]

        return compute_distribution(
            values=values,
            bin_edges=bin_edges,
            prior_weights=self.weights,
            classification_weights=self.get_mask(zl_index, color_condition="all"),
            normalize=normalize,
            missing_data=self.missing_data,
            CDZs=self.CDZs if pdz_mode else None,
            PDZ_x=self.PDZ_x if pdz_mode else None,
        )

    def write(
        self,
        output_dir,
        file_name,
        reference_sample_name="",
        notes="",
        write_metrics=True,
        write_redshift_distributions=False,
        n_bins_redshift=200,
        min_redshift=0.0,
        max_redshift=4.0,
    ):
        """
        Save the optimal color selection.

        The dictionary of optimal cuts is saved to a YAML file.
        Metrics and redshift distributions are saved to CSV files.

        Parameters
        ----------
        output_dir : str
            Path to the directory where the file will be saved.
        file_name : str
            Base name of the output file (without extension).
            All file names start with "FOMO-Z_".
        reference_sample_name : str, default=""
            Name of the reference sample, stored in the
            ``reference_sample`` entry of the dictionary.
        notes : str, default=""
            Additional notes entering the metadata.
        write_metrics : bool, default=True
            If True, write the `FOMO-Z_{file_name}_metrics.csv` file containing
            purity, completeness, and failure rate
            as a function of limiting redshift, and their errors.
        write_redshift_distributions : bool, default=False
            If True, write the
            `FOMO-Z_{file_name}_redshift_distributions.csv` file,
            containing the redshift distributions of color-selected
            reference objects for each limiting redshift in
            `SelectionDefinition`.

            Distributions are computed from reference point-estimate
            redshifts (if PDZs are not provided) and weighted by the prior weights supplied
            to `Dataset`. If PDZs were provided to dataset, they are used to
            compute the histograms rather than using redshift point estimates.

            The first three columns contain the lower bin edges,
            upper bin edges, and bin centers. Subsequent columns,
            named `z_lim_{i}`, contain the probability densities for
            the corresponding limiting-redshift indices.

            All histograms share the same bins and are normalized
            to unit integral over [min_redshift, max_redshift].
            An error is raised if a histogram has zero total weight
            within this interval.
        n_bins_redshift : int, default=200
            Number of redshift histogram bins. Used only if `write_redshift_distributions`
            is True.
        min_redshift : float, default=0.0
            Minimum redshift. Used only if `write_redshift_distributions` is True.
        max_redshift : float, default=4.0
            Maximum redshift. Used only if `write_redshift_distributions` is True.

        Raises
        ------
        ValueError
            If calibration has not been run.

        Notes
        -----
        The output directory is created if needed. Existing files with the
        same names are overwritten. The YAML stores metadata and spline
        parameters for use by Classifier; per-threshold c_optimal and
        s_optimal arrays are omitted from the exported copy. The in-memory
        dictionary is unchanged. Iteration histories are not exported.

        Metrics columns are z_lim, C, P, F, C_std, P_std, and F_std.
        Uncertainties are NaN when bootstrap was disabled. The YAML and
        metrics are written before optional redshift distributions, so a
        later distribution error can leave these earlier files in place.
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
                hist = self.get_distribution(i, "redshift", bin_edges, normalize=True)
                dist_df[f"z_lim_{i}"] = hist

            file_path = os.path.join(
                output_dir, f"FOMO-Z_{file_name}_redshift_distributions.csv"
            )
            dist_df.to_csv(file_path, index=False)

    def get_keys(self):
        """
        Return the calibrated color-cut expressions.

        Returns
        -------
        list of str
            Expression keys in dict, excluding ``"metadata"``. Each key
            can be passed to get_mask as color_condition.

        Raises
        ------
        ValueError
            If the sum of calibrated completeness values is zero; normally
            indicates that no successful calibration is available.
        """
        if np.sum(self.C) == 0.0:
            raise ValueError("No calibration is available! Run process() first.")

        return [key for key in list(self.dict.keys()) if key != "metadata"]

    def get_mask(self, zl_index, color_condition="all"):
        """
        Evaluate calibrated color cuts on the reference sample.

        Parameters
        ----------
        zl_index : int
            Index into ``SelectionDefinition.z_lim``; this is an index, not
            a redshift value. This method uses stored optimal parameters,
            rather than evaluating the parameter splines at arbitrary redshift.
        color_condition : str, default="all"
            ``"all"`` combines the expressions with logical OR. Otherwise,
            supply one expression returned by get_keys.

        Returns
        -------
        numpy.ndarray of bool, shape (n_reference,)
            Objects passing at least one requested color cut. This is a
            photometric selection mask, not the target-membership mask:
            contaminants can be selected. No object weights are included.

        Raises
        ------
        AssertionError
            If color_condition is not a string (when assertions are enabled).
        ValueError
            If calibration is unavailable or the expression is unknown.
        IndexError
            If zl_index is outside the threshold array.

        Notes
        -----
        A cut is evaluated only when the requested threshold appears among
        its unique stored spline knots. The corresponding entry in c_optimal
        and, where present, s_optimal is used. Comparisons involving NaN
        colors do not pass; another cut using available bands can still
        select the object. Use Classifier for independent test samples.
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
