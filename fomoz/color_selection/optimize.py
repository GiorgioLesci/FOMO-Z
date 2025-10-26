"""
OPTIMIZE COLOR CUT COMBINATION.

This module selects the optimal color conditions and the corresponding
optimal parameters, given any lower limit in redshift.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import os

import numpy as np
import multiprocessing as mp
import copy
from tqdm import tqdm
import time
import yaml
import matplotlib.pyplot as plt

from .auxiliary import (
    set_plot,
    make_yaml_serializable,
    float_representer,
    build_nested_dict,
    restructure_dict,
    interpolate_parameters,
)

# Register the custom float representer for both default and safe dumpers
yaml.add_representer(float, float_representer)
yaml.add_representer(float, float_representer, Dumper=yaml.SafeDumper)


class Optimize:
    """
    Class for optimizing color cuts based on metrics computed by `Measure`.

    This class takes the results from a `Measure` instance (which stores
    metrics for a grid of color cuts) and provides structures for ranking,
    aggregating, and optimizing those cuts.

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
    dict_single : dict
        Container for storing results based on a single purity threshold.
    dict_iterative : dict
        Container for storing results based on an iterative purity threshold.
    P_single_tot : numpy.ndarray
        Total purity values as a function of `z_lim` (single threshold).
    C_single_tot : numpy.ndarray
        Total completeness values as a function of `z_lim` (single threshold).
    F_single_tot : numpy.ndarray
        Total failure rates (single threshold).
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

    Notes
    -----
    - The condition *F < 1* is an additional requirement applied in the
      optimization logic (see methods of this class).
    """

    def __init__(self, measure):
        self.bands = measure.bands
        self.colors = measure.colors
        self.dataset = measure.dataset
        self.z_lim = measure.z_lim
        self.dict_all = measure.dict_all

        self.dict_single = {}
        self.dict_iterative = {}
        self.dict_iterative_steps = {}
        self.P_single_tot = np.zeros(len(self.z_lim))
        self.C_single_tot = np.zeros(len(self.z_lim))
        self.F_single_tot = np.zeros(len(self.z_lim))
        self.P_iterative_tot = np.zeros(len(self.z_lim))
        self.C_iterative_tot = np.zeros(len(self.z_lim))
        self.F_iterative_tot = np.zeros(len(self.z_lim))

    def _get_optimal_single_threshold_cuts(self, purity_min, completeness_min):
        """
        Identify and store the optimal single-purity-threshold cuts that
        maximize single cut completeness.

        This method analyzes all color cuts stored in `self.dict_all`
        and selects those whose purity exceeds `purity_min`. For each
        valid cut, only the parameter configuration that maximizes completeness
        (subject to the purity constraint) is kept. The results are
        stored in `self.dict_single`.

        Parameters
        ----------
        purity_min : float or array-like
            Minimum required purity. Only cuts with purity
            greater than this value are considered.
        completeness_min : float
            Minimum required completeness threshold. Cuts below this
            threshold are discarded even if they satisfy the purity
            requirement.

        Returns
        -------
        None
            The method updates `self.dict_single` with the optimal cuts
            but does not return a value.
        """
        if completeness_min > 1.0 or completeness_min < 0.0:
            raise ValueError("completeness_min must range within [0,1]")

        for z_l, cuts_by_magpair in self.dict_all.items():

            P_min = purity_min[np.where(self.z_lim == z_l)[0][0]]
            if P_min > 1.0 or P_min < 0.0:
                raise ValueError("The minimum purity must range within [0,1]")

            self.dict_single[z_l] = {}

            for base_pair, sub_cuts in cuts_by_magpair.items():
                self.dict_single[z_l][base_pair] = {}

                for comp_pair, metrics in sub_cuts.items():

                    # ----- Cross-color cuts -----
                    if "s" in metrics and "c" in metrics:

                        best_conditions = [None] * 4
                        max_completeness = [-np.inf] * 4

                        for j in range(len(metrics["TP"])):  # j = index in (s, c) grid
                            for i in range(4):  # i = condition index (0 to 3)
                                purity = (
                                    metrics["TP"][j][i]
                                    / (metrics["TP"][j][i] + metrics["FP"][j][i])
                                    if metrics["TP"][j][i] > 0
                                    else 0.0
                                )
                                completeness = (
                                    metrics["TP"][j][i]
                                    / (metrics["TP"][j][i] + metrics["FN"][j][i])
                                    if metrics["TP"][j][i] > 0
                                    else 0.0
                                )
                                f_low = (
                                    metrics["FP"][j][i]
                                    / (metrics["FP"][j][i] + metrics["TN"][j][i])
                                    if metrics["FP"][j][i] > 0
                                    else 0.0
                                )

                                if (
                                    purity > P_min
                                    and completeness > completeness_min
                                    and f_low < 1.0
                                ):
                                    if completeness > max_completeness[i]:
                                        max_completeness[i] = completeness
                                        best_conditions[i] = {
                                            "s": float(metrics["s"][j]),
                                            "c": float(metrics["c"][j]),
                                            "condition_index": i,
                                            "TP": metrics["TP"][j][i],
                                            "FP": metrics["FP"][j][i],
                                            "FN": metrics["FN"][j][i],
                                            "TN": metrics["TN"][j][i],
                                        }

                        # Remove None entries
                        # (in case some conditions didn't pass the threshold)
                        best_conditions = [
                            cond for cond in best_conditions if cond is not None
                        ]

                        if best_conditions:
                            self.dict_single[z_l][base_pair][
                                comp_pair
                            ] = best_conditions

                    # ----- Simple (1-color) cuts -----
                    else:

                        best_conditions = []
                        max_comp_up = -np.inf
                        max_comp_low = -np.inf
                        best_up = None
                        best_low = None

                        for i, c in enumerate(metrics["c"]):
                            purity_up = (
                                metrics["TP"][i][0]
                                / (metrics["TP"][i][0] + metrics["FP"][i][0])
                                if metrics["TP"][i][0] > 0
                                else 0.0
                            )
                            purity_low = (
                                metrics["TP"][i][1]
                                / (metrics["TP"][i][1] + metrics["FP"][i][1])
                                if metrics["TP"][i][1] > 0
                                else 0.0
                            )
                            comp_up = (
                                metrics["TP"][i][0]
                                / (metrics["TP"][i][0] + metrics["FN"][i][0])
                                if metrics["TP"][i][0] > 0
                                else 0
                            )
                            comp_low = (
                                metrics["TP"][i][1]
                                / (metrics["TP"][i][1] + metrics["FN"][i][1])
                                if metrics["TP"][i][1] > 0
                                else 0
                            )
                            f_low_up = (
                                metrics["FP"][i][0]
                                / (metrics["FP"][i][0] + metrics["TN"][i][0])
                                if metrics["FP"][i][0] > 0
                                else 0
                            )
                            f_low_low = (
                                metrics["FP"][i][1]
                                / (metrics["FP"][i][1] + metrics["TN"][i][1])
                                if metrics["FP"][i][1] > 0
                                else 0
                            )

                            # Upper cut (color > c)
                            if (
                                purity_up > P_min
                                and comp_up > completeness_min
                                and f_low_up < 1.0
                            ):
                                if comp_up > max_comp_up:
                                    max_comp_up = comp_up
                                    best_up = {
                                        "c": float(c),
                                        "condition_index": 0,
                                        "TP": metrics["TP"][i][0],
                                        "FP": metrics["FP"][i][0],
                                        "FN": metrics["FN"][i][0],
                                        "TN": metrics["TN"][i][0],
                                    }

                            # Lower cut (color < c)
                            if (
                                purity_low > P_min
                                and comp_low > completeness_min
                                and f_low_low < 1.0
                            ):
                                if comp_low > max_comp_low:
                                    max_comp_low = comp_low
                                    best_low = {
                                        "c": float(c),
                                        "condition_index": 1,
                                        "TP": metrics["TP"][i][1],
                                        "FP": metrics["FP"][i][1],
                                        "FN": metrics["FN"][i][1],
                                        "TN": metrics["TN"][i][1],
                                    }

                        if best_up or best_low:
                            self.dict_single[z_l][base_pair][comp_pair] = []
                            if best_up:
                                self.dict_single[z_l][base_pair][comp_pair].append(
                                    best_up
                                )
                            if best_low:
                                self.dict_single[z_l][base_pair][comp_pair].append(
                                    best_low
                                )

    def _process_iterative_for_one_z(self, args):
        """
        Optimize the color cut selection for a single redshift threshold `z_l`.

        For the given redshift value, this method searches for the set of
        cuts that ($\mathcal{P}_{\rm iter}^{\rm ini}$: `start_purity`,
        $\delta\mathcal{P}_{\rm iter}$: `step`, $z_{\rm l}$: `z_l`,
        $\mathcal{P}_{\rm min}$: `purity_min`):
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

        Parameters
        ----------
        args : tuple
            A 4-element tuple containing:
            - z_l : float
                The redshift threshold being processed.
            - purity_tot_min : float
                Minimum required total purity for the optimized set of cuts.
            - start_purity : float
                Starting value for the individual purity threshold search.
            - step : float
                Increment step applied when scanning over purity thresholds.
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
        z_l, purity_tot_min, start_purity, step, store_iterations = args

        if start_purity <= purity_tot_min:
            raise ValueError(
                f"The starting purity, {start_purity},"
                f" is <= than the global purity threshold, {purity_tot_min}."
            )

        self.dict_iterative[z_l] = {}
        all_cuts = self.dict_all.get(z_l, {})
        if not all_cuts:
            return z_l, None

        idx = np.where(self.z_lim == z_l)[0][0]
        redshift = self.dataset["redshift"].values.astype(np.float32)

        # -- STEP 1: Build a container of all canidate color cuts
        candidate_pool = []
        for base_pair, comp_cuts in all_cuts.items():
            for comp_pair, metrics in comp_cuts.items():
                if comp_pair == base_pair:  # simple
                    for i, c_val in enumerate(metrics["c"]):
                        for cond_index in (0, 1):
                            TP = metrics["TP"][i][cond_index]
                            FP = metrics["FP"][i][cond_index]
                            FN = metrics["FN"][i][cond_index]
                            TN = metrics["TN"][i][cond_index]
                            purity = TP / (TP + FP) if TP + FP > 0 else 0.0
                            compl = TP / (TP + FN) if TP + FN > 0 else 0.0
                            f_low = FP / (FP + TN) if FP + TN > 0 else 0.0
                            candidate_pool.append(
                                {
                                    "base_pair": base_pair,
                                    "comp_pair": comp_pair,
                                    "c": float(c_val),
                                    "condition_index": cond_index,
                                    "purity": purity,
                                    "completeness": compl,
                                    "f_low": f_low,
                                }
                            )
                else:  # cross
                    for i in range(len(metrics["s"])):
                        for cond_index in range(4):
                            TP = metrics["TP"][i][cond_index]
                            FP = metrics["FP"][i][cond_index]
                            FN = metrics["FN"][i][cond_index]
                            TN = metrics["TN"][i][cond_index]
                            purity = TP / (TP + FP) if TP + FP > 0 else 0.0
                            compl = TP / (TP + FN) if TP + FN > 0 else 0.0
                            f_low = FP / (FP + TN) if FP + TN > 0 else 0.0
                            candidate_pool.append(
                                {
                                    "base_pair": base_pair,
                                    "comp_pair": comp_pair,
                                    "s": float(metrics["s"][i]),
                                    "c": float(metrics["c"][i]),
                                    "condition_index": cond_index,
                                    "purity": purity,
                                    "completeness": compl,
                                    "f_low": f_low,
                                }
                            )

        # -- STEP 2: Iterative process
        current_thresh = start_purity
        best_by_key = {}
        dict_iterative_steps = {}

        C_tot = np.zeros_like(self.z_lim, dtype=np.float32)
        P_tot = np.zeros_like(self.z_lim, dtype=np.float32)
        F_tot = np.zeros_like(self.z_lim, dtype=np.float32)

        iteration = 0
        while current_thresh > 0:
            # Get the colour conditions that satisfy the
            # P, C, and F conditions and select the parameters
            # of such conditions that maximize C
            new_best_by_key = {}
            C_tot_new = np.zeros_like(self.z_lim, dtype=np.float32)
            P_tot_new = np.zeros_like(self.z_lim, dtype=np.float32)
            F_tot_new = np.zeros_like(self.z_lim, dtype=np.float32)

            for cut in candidate_pool:
                if cut["purity"] < current_thresh or cut["f_low"] >= 1.0:
                    continue  # skip this cut
                key = (
                    cut["base_pair"],
                    cut["comp_pair"],
                    cut["condition_index"],
                )
                if (
                    key not in new_best_by_key
                    or cut["completeness"] > new_best_by_key[key]["completeness"]
                ):
                    new_best_by_key[key] = cut

            selection = list(new_best_by_key.values())
            nested = build_nested_dict(selection)
            self._combine_optimal_cuts({z_l: nested}, C_tot_new, P_tot_new, F_tot_new)

            if P_tot_new[idx] >= purity_tot_min:

                best_by_key = new_best_by_key
                C_tot = C_tot_new
                P_tot = P_tot_new
                F_tot = F_tot_new
                current_thresh -= step

                if store_iterations:
                    dict_iterative_steps[iteration] = nested

            else:
                break  # below the threshold

            iteration += 1

        return z_l, list(best_by_key.values()), dict_iterative_steps

    def _combine_optimal_cuts(self, dictionary, C_tot, P_tot, F_tot):
        """
        Combine multiple optimal color cuts and compute
        aggregate performance metrics.

        This method aggregates purity, completeness, and foreground
        failure rates across a set of optimal cuts stored in
        `dictionary`. The combined results are accumulated into the
        provided arrays `C_tot`, `P_tot`, and `F_tot`.

        Parameters
        ----------
        dictionary : dict
            Dictionary of selected optimal cuts (e.g., `self.dict_single`
            or `self.dict_iterative`), where each entry contains the metrics
            for a cut.
        C_tot : numpy.ndarray
            Array that accumulates total completeness values across cuts.
        P_tot : numpy.ndarray
            Array that accumulates total purity values across cuts.
        F_tot : numpy.ndarray
            Array that accumulates total foreground failure rate values
            across cuts.

        Returns
        -------
        None
            Updates the input arrays (`C_tot`, `P_tot`, `F_tot`) in place
            but does not return a value.
        """
        redshift = self.dataset["redshift"].values.astype(np.float32)

        for z_l in self.z_lim:
            if z_l not in dictionary:
                continue
            mask_total = np.zeros_like(redshift, dtype=bool)

            for base_pair, comp_cuts in dictionary.get(z_l, {}).items():
                for comp_pair, conditions in comp_cuts.items():
                    for cond in conditions:
                        mag1, mag2 = base_pair
                        if comp_pair == base_pair:
                            # Simple cut
                            color = self.dataset[mag1].values.astype(
                                np.float32
                            ) - self.dataset[mag2].values.astype(np.float32)
                            if cond["condition_index"] == 0:  # color > c
                                mask = color > cond["c"]
                            else:  # color < c
                                mask = color < cond["c"]
                        else:
                            # Cross cut
                            mag3, mag4 = comp_pair
                            color1 = self.dataset[mag1].values.astype(
                                np.float32
                            ) - self.dataset[mag2].values.astype(np.float32)
                            color2 = self.dataset[mag3].values.astype(
                                np.float32
                            ) - self.dataset[mag4].values.astype(np.float32)
                            s = cond["s"]
                            c = cond["c"]
                            idx = cond["condition_index"]

                            if idx == 0:
                                mask = color1 > s * color2 + c
                            elif idx == 1:
                                mask = color1 > -s * color2 + c
                            elif idx == 2:
                                mask = color1 < s * color2 + c
                            elif idx == 3:
                                mask = color1 < -s * color2 + c
                            else:
                                continue

                        mask_total |= mask  # OR logic

            # Compute metrics
            selected = mask_total
            redshift_sel = redshift[selected]
            redshift_unsel = redshift[~selected]

            TP = np.sum(redshift_sel > z_l)
            FP = np.sum(redshift_sel < z_l)
            FN = np.sum(redshift_unsel > z_l)
            TN = np.sum(redshift_unsel < z_l)

            idx = np.where(self.z_lim == z_l)[0][0]
            C_tot[idx] = TP / (TP + FN) if TP + FN > 0 else 0.0
            P_tot[idx] = TP / (TP + FP) if TP + FP > 0 else 0.0
            F_tot[idx] = FP / (FP + TN) if FP + TN > 0 else 0.0

    def process_L24(self, purity_min, completeness_min=0.0, smoothing=0.0, k=3):
        """
        Derives the optimal color conditions based on a single purity threshold.
        Implements the methods detailed in Lesci et al. (2024, A&A, 684, A139),
        available at:
        https://ui.adsabs.harvard.edu/abs/2024A%26A...684A.139E/abstract

        This method
        1. identifies all color cuts that satisfy the given minimum purity and
           completeness thresholds when evaluated individually,
        2. selects optimal single cut, namely those with parameter values
           that maximize the individual cut completeness,
        3. combines these optimal single cuts into an overall set of conditions,
           computing the resulting total purity, completeness, and failure rate,
        4. restructures the resulting dictionary of cuts into expression-based keys, and
        5. interpolates the cut parameters across redshift bins.

        Parameters
        ----------
        purity_min : float or array-like
            Minimum purity threshold applied to each color cut. If float, the
            same threshold is applied at any limiting redshift point.
            If array-like, a different threshold is used at each limiting
            redshift.
        completeness_min : float, optional
            Minimum completeness threshold applied to each color cut.
            Default is 0.
        smoothing : float, optional
            Smoothing factor passed to `UnivariateSpline` during interpolation.
            Default is 0.0 (interpolating spline).
        k : int, optional
            Requested degree of the spline interpolation. Default is 3 (cubic).
            The actual degree is adjusted case by case:
            - If there are fewer than ``k+1`` points, the degree is reduced
              to ``n_points - 1``.
            - If there is only one point, interpolation is skipped.

        Returns
        -------
        None
            The method updates the following instance attributes in place:
            - ``self.dict_single`` : dict
                Dictionary storing the optimal single-threshold cuts.
            - ``self.P_single_tot`` : numpy.ndarray
                Total purity across all redshift bins.
            - ``self.C_single_tot`` : numpy.ndarray
                Total completeness across all redshift bins.
            - ``self.F_single_tot`` : numpy.ndarray
                Total foreground failure rate across all redshift bins.
        """
        print(
            "\nDeriving the optimal cuts in all color-color spaces",
            "(single purity threshold case)...",
        )
        if isinstance(purity_min, float):
            purity_min = [purity_min] * len(self.z_lim)
        self._get_optimal_single_threshold_cuts(purity_min, completeness_min)

        print("Combining the optimal cuts...")
        self._combine_optimal_cuts(
            self.dict_single,
            self.C_single_tot,
            self.P_single_tot,
            self.F_single_tot,
        )
        print("Done!")
        print("\nCompleteness:", self.C_single_tot)
        print("Purity:", self.P_single_tot)
        print("Foreground failure rate:", self.F_single_tot)

        restructure_dict(
            self.dict_single, self.bands, self.colors, self.z_lim, purity_min
        )
        interpolate_parameters(self.dict_single, smoothing=smoothing, k=k)

    def process(
        self,
        purity_tot_min,
        start_purity=0.999,
        step=0.001,
        cpu_number=None,
        smoothing=0.0,
        k=3,
        store_iterations=False,
    ):
        """
        Iterative search for optimal color conditions.

        This method incrementally relaxes the single cut purity threshold in
        order to identify a set of cuts that maximize total completeness while
        still satisfying the global purity constraint. The algorithm proceeds
        iteratively, starting from a strict initial purity threshold and reducing
        it by `step` until the total purity remains above `purity_tot_min`.

        Constraints
        -----------
        For each redshift value, this method searches for the set of
        cuts that ($\mathcal{P}_{\rm iter}^{\rm ini}$: `start_purity`,
        $\delta\mathcal{P}_{\rm iter}$: `step`, $z_{\rm l}$: `z_l`,
        $\mathcal{P}_{\rm min}$: `purity_min`):
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
        purity_tot_min : float or array-like
            Required minimum global purity across all redshift bins. If float,
            the same threshold is applied at any limiting redshift point.
            If array-like, a different threshold is used at each limiting
            redshift.
        start_purity : float, optional
            Initial strict threshold for the individual cut purities.
            Must be greater than `purity_tot_min`. Default is 0.999.
        step : float, optional
            Increment by which the single-cut purity threshold is reduced
            during the search. Default is 0.001.
        cpu_number : int or None, optional
            Number of CPU cores to use for parallel processing. If None,
            defaults to using all available cores minus two.
        smoothing : float, optional
            Smoothing factor passed to `UnivariateSpline` during interpolation.
            Default is 0.0 (interpolating spline).
        k : int, optional
            Requested degree of the spline interpolation. Default is 3 (cubic).
            The actual degree is adjusted case by case:
            - If there are fewer than ``k+1`` points, the degree is reduced
              to ``n_points - 1``.
            - If there is only one point, interpolation is skipped.
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
                Total foreground failure rate across all bins.
        """
        if isinstance(purity_tot_min, float):
            purity_tot_min = [purity_tot_min] * len(self.z_lim)

        print("\nRunning iterative purity threshold search...")
        start_time = time.time()

        # Parallel processing
        N_cpu = (
            min(cpu_number, mp.cpu_count() - 2)
            if cpu_number
            else max(mp.cpu_count() - 2, 1)
        )
        args_list = [
            (
                z_l,
                purity_tot_min[np.where(self.z_lim == z_l)[0][0]],
                start_purity,
                step,
                store_iterations,
            )
            for z_l in self.z_lim
        ]
        with mp.Pool(processes=N_cpu) as pool:
            results = pool.map(self._process_iterative_for_one_z, args_list)

        for result in results:
            z_l, cut_list, steps = result
            self.dict_iterative[z_l] = build_nested_dict(cut_list)
            self.dict_iterative_steps[z_l] = steps

        print("\nCombining the final optimal cuts across redshift bins...")
        self._combine_optimal_cuts(
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
        print("\nCompleteness:", self.C_iterative_tot)
        print("Purity:", self.P_iterative_tot)
        print("Foreground failure rate:", self.F_iterative_tot)

        restructure_dict(
            self.dict_iterative, self.bands, self.colors, self.z_lim, purity_tot_min
        )
        interpolate_parameters(self.dict_iterative, smoothing=smoothing, k=k)

    def write_optimal_selection(
        self,
        output_dir,
        file_name,
        optimization_type="iterative",
        reference_sample_name="",
        notes="",
    ):
        """
        Save the optimal color selection to a YAML file.

        Depending on the chosen optimization type, either the dictionary
        of single-threshold cuts or the dictionary of iterative-threshold
        cuts is serialized and written to file.

        Parameters
        ----------
        output_dir : str
            Path to the directory where the file will be saved.
        file_name : str
            Base name of the output file (without extension).
        optimization_type : {"single", "iterative"}, optional
            Type of optimization results to save:
            - "single": saves ``self.dict_single``
            - "iterative": saves ``self.dict_iterative``
            Default is "iterative".
        reference_sample_name : str, optional
            Name of the reference sample, stored in the
            ``reference_sample`` entry of the dictionary.
        notes : str, optional
            Additional notes entering the metadata.

        Returns
        -------
        None
            The selected dictionary is saved as a ``.yaml`` file.
        """
        if optimization_type not in ["single", "iterative"]:
            raise ValueError(f"Optimization type '{optimization_type}' is not allowed.")

        if optimization_type == "single":
            dictionary = self.dict_single
        else:
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

    def plot(self, loclegend="upper right"):
        """
        Plot the performance metrics for the optimized color cuts.

        This method produces a three-panel figure showing:
        1. Completeness as a function of redshift threshold.
        2. Purity as a function of redshift threshold.
        3. Foreground failure rate as a function of redshift threshold.

        If results from both optimization methods are available, the curves are
        plotted for:
        - Single-threshold optimization (dashed red lines).
        - Iterative-threshold optimization (solid black lines).

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

        if np.sum(self.C_single_tot) != 0.0:
            ax[0].plot(
                self.z_lim,
                self.C_single_tot * 100.0,
                color="r",
                linestyle="--",
                label="Single threshold",
            )
            ax[1].plot(self.z_lim, self.P_single_tot * 100.0, color="r", linestyle="--")
            ax[2].plot(self.z_lim, self.F_single_tot * 100.0, color="r", linestyle="--")

        if np.sum(self.C_iterative_tot) != 0.0:
            ax[0].plot(
                self.z_lim,
                self.C_iterative_tot * 100.0,
                color="k",
                linestyle="-",
                label="Iterative threshold",
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

        ax[0].legend(loc=loclegend, fontsize=13)
        plt.show()
