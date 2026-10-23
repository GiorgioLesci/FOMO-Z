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
    """Calibrate linear color selections."""

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
        r"""
        Calibrate optimal linear color selections.

        This class selects objects occupying the outskirts of color-color
        distributions, as is often the case for background galaxy populations.
        For more general selections, see
        :class:`~fomoz.binary_classification.grid_classification.GridClassification`.

        For each pair of colors :math:`(x, y)`, the following conditions
        are considered:

        .. math::

            \begin{aligned}
                x &> c_1, & x &< c_2, \\
                y &> c_3, & y &< c_4, \\
                x &> s_1 y + c_5, & x &> s_2 y + c_6, \\
                x &< s_3 y + c_7, & x &< s_4 y + c_8.
            \end{aligned}

        Here, :math:`c_1,\ldots,c_8` are real-valued parameters,
        :math:`s_1,s_3>0`, and :math:`s_2,s_4<0`.
        All conditions, both within and across color-color spaces,
        are combined through the logical OR operator, :math:`\lor`.

        The algorithm progressively relaxes the purity threshold applied
        to individual cuts to increase global completeness while maintaining
        the required global purity. Bootstrap uncertainties on the resulting
        metrics can also be estimated.

        For each limiting redshift specified in ``selection_definition``,
        let :math:`\delta\mathcal{P}_{\rm step}` denote ``step``,
        :math:`\mathcal{C}_{\rm min}` denote ``completeness_min``, and
        :math:`\mathcal{P}_{\rm min}` denote ``purity_min``.
        The optimization proceeds as follows:

        1. Progressively lower the single-cut purity threshold from unity
           in increments of :math:`\delta\mathcal{P}_{\rm step}`.
           After :math:`n` decrements, the threshold is

           .. math::

               \mathcal{P}_{\rm iter}^{n}
               = 1 - n\,\delta\mathcal{P}_{\rm step}.

        2. For each color pair, retain a candidate condition if at least
           one parameter combination satisfies both
           :math:`\mathcal{P}^{j}>\mathcal{P}_{\rm iter}^{n}` and
           :math:`\mathcal{C}^{j}>\mathcal{C}_{\rm min}`.
           Here, :math:`j` indexes the candidate conditions, while
           :math:`\mathcal{P}^{j}` and :math:`\mathcal{C}^{j}` are their
           individual purity and completeness.

        3. For each retained condition, choose the admissible parameter
           combination that maximizes :math:`\mathcal{C}^{j}`.

        4. Combine the resulting cuts across all color-color spaces using
           :math:`\lor`, and evaluate the global purity :math:`\mathcal{P}`
           and completeness :math:`\mathcal{C}`.

        5. Continue relaxing the single-cut threshold while the combined
           selection satisfies :math:`\mathcal{P}>\mathcal{P}_{\rm min}`,
           retaining the optimal admissible cuts.

        The calibrated cuts are stored in an expression-based dictionary,
        together with global purity, completeness, and failure rate metrics.
        Cut parameters are interpolated across limiting redshifts to
        provide continuous selections.

        Individual conditions with a failure rate equal to one are
        always excluded.

        Parameters
        ----------
        dataset : Dataset
            Reference dataset containing photometry, target properties,
            and band information.
        selection_definition : SelectionDefinition
            Target selection criteria, such as redshift thresholds.
        purity_min : float or array-like
            Required minimum global purity at each limiting redshift.
            If a scalar, the same threshold is applied at every limiting
            redshift. If array-like, provide one threshold per limiting
            redshift.
        completeness_min : float, default=0.05
            Minimum completeness required for an individual color cut.
            This constraint applies to single cuts, not to the combined
            selection. It helps exclude cuts supported by very few target
            objects and reduces sensitivity to statistical fluctuations.
        threshold_buffer : float, default=0.05
            Buffer subtracted from ``purity_min`` when retaining candidate
            color cuts. Conditions not satisfying
            ``purity_min - threshold_buffer`` are discarded before
            optimization to reduce memory usage.
            The default corresponds to five percentage points.
            An error is raised if the buffer is too small for the
            optimization to proceed.
        s_samples : array-like, default=np.linspace(0.3, 2.5, 50)
            Strictly positive slope magnitudes used to generate inclined
            cuts. Both positive and negative slopes are considered.
            The default provides 50 equally spaced values between
            0.3 and 2.5, including both endpoints.
        c_precision : float, default=0.05
            Sampling step for the parameters :math:`c_i`.
            Smaller values provide finer sampling at greater computational
            cost. Halving this step approximately doubles the number of
            intercept samples per slope, with a corresponding increase
            in runtime and memory usage.
        k : int, default=1
            Requested spline degree for interpolating cut parameters.
            The default gives linear interpolation. If fewer than
            ``k + 1`` points are available, the degree is reduced to
            ``n_points - 1``. Interpolation is skipped when only one
            point is available.
        step : float, default=0.0001
            Decrement of the single-cut purity threshold at each
            optimization iteration,
            :math:`\delta\mathcal{P}_{\rm step}`.
        do_bootstrap : bool, default=True
            If True, estimate uncertainties on completeness, purity,
            and failure rate through bootstrap resampling.
        n_bootstraps : int, default=1000
            Number of bootstrap resamples used to estimate uncertainties.
        verbose : bool, default=True
            If True, print progress messages and computation times.

        Attributes
        ----------
        dict : dict
            Calibrated linear color cuts and associated results.
        P : numpy.ndarray
            Global purity at each limiting redshift.
        C : numpy.ndarray
            Global completeness at each limiting redshift.
        F : numpy.ndarray
            Global failure rate at each limiting redshift.
        P_std : numpy.ndarray
            Bootstrap uncertainty on global purity.
        C_std : numpy.ndarray
            Bootstrap uncertainty on global completeness.
        F_std : numpy.ndarray
            Bootstrap uncertainty on global failure rate.
        attained_threshold : numpy.ndarray
            Single-cut purity threshold attained during optimization
            at each limiting redshift.
        """
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
                raise ValueError(
                    "purity_min accepted values lie in the range [0, 1)."
                )

        maximized_quantity="completeness"  # hard-coded for now
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
                errMsg += " To check this, try to set threshold_buffer equal to purity_min."
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
        r"""
        Perform the iterative search for optimal linear color conditions.

        Here, either completeness or purity are maximized, depending on what was
        set in the `SelectionDefinition` instance given in input to the constructor of this class.

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
                hist = self.get_distribution(i, "redshift", bin_edges, normalize=True)
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
            Index of the redshift threshold in SelectionDefinition.z_lim.
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
