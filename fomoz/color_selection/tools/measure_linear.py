"""
Auxiliary class to measure linear color cut performance.

This module provides the computation of linear color condition metrics,
given any limit on galaxy properties and color condition parameter values.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

from .color_domain import c_domain, c_domain_given_s
from .statistics import compute_for_cross, compute_for_simple
from .variable_handler import set_input_variable_precision

# Globals visible to workers
GLOBAL_color1 = None
GLOBAL_color2 = None
GLOBAL_shared_mask = None


def init_worker(shared_c1, shared_c2, shared_target_mask):
    """
    Initialize global data shared across worker processes using shared memory.

    This function reconstructs numpy arrays from shared memory buffers
    to avoid data copying between processes.

    Parameters
    ----------
    shared_c1 : multiprocessing.sharedctypes.SynchronizedArray or None
        Shared memory array for the first color data. Use None if not available.
    shared_c2 : multiprocessing.sharedctypes.SynchronizedArray or None
        Shared memory array for the second color data. Use None if not available.
    shared_target_mask : multiprocessing.sharedctypes.SynchronizedArray
        Shared memory array for galaxy redshift values.
    """
    global GLOBAL_color1, GLOBAL_color2, GLOBAL_shared_mask
    GLOBAL_color1 = (
        np.frombuffer(shared_c1.get_obj(), dtype=np.float32) if shared_c1 else None
    )
    GLOBAL_color2 = (
        np.frombuffer(shared_c2.get_obj(), dtype=np.float32) if shared_c2 else None
    )
    bool_array = np.frombuffer(shared_target_mask.get_obj(), dtype=np.int8)
    GLOBAL_shared_mask = bool_array.astype(bool)


def compute_for_simple_worker(args):
    """
    Wrap `compute_for_simple` using global data set by `init_worker`.

    Parameters are passed via `args` to remain compatible with `pool.map`.
    See `compute_for_simple` for full documentation.
    """
    global GLOBAL_color1, GLOBAL_shared_mask
    c = args
    return compute_for_simple((c, GLOBAL_color1, GLOBAL_shared_mask))


def compute_for_cross_worker(args):
    """
    Wrap `compute_for_cross` using global data set by `init_worker`.

    Parameters are passed via `args` to remain compatible with `pool.map`.
    See `compute_for_cross` for full documentation.
    """
    global GLOBAL_color1, GLOBAL_color2, GLOBAL_shared_mask
    s, c = args
    return compute_for_cross((s, c, GLOBAL_color1, GLOBAL_color2, GLOBAL_shared_mask))


class MeasureLinear:
    """Measure the metrics of color conditions."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        colors,
        target_definition,
        thresholded_quantity,
        maximized_quantity_min,
        buffered_threshold,
        s_samples,
        c_precision,
        step,
    ):
        """
        Measure the metrics of color conditions.

        Class defining color selection completeness, purity, and
        failure rate for all color conditions, given any
        redshift limit values and color condition parameters.

        This class standardizes input photometric data, handles missing values,
        constructs color arrays from provided bands, and computes color selection
        metrics. These selections are defined as linear relations between
        colors and are evaluated over a grid of slope–intercept combinations.

        Parameters
        ----------
        dataset : pandas.DataFrame
            Dataframe containing the dataset.
        colors : list
            List of colors.
        target_definition : TargetDefinition
            A `TargetDefinition` object containing target selections (e.g. redshift thresholds).
        thresholded_quantity : str
            The quantity to on which the thresholding is applied.
            Possibilities are either "completeness" or "purity".
        maximized_quantity_min : float
            Required minimum for the maximized completeness (if `maximized_quantity == "completeness"`)
            or purity (if `maximized_quantity == "purity"`) across all redshift bins.
        buffered_threshold : float
            Buffered threshold on the thresholded quantity (see the documentation of `CalibrateLinear`).
        s_samples : array-like
            Array of slope values for generating linear cuts in color-color space.
            These samples must be strictly positive.
        c_precision : float
            Precision of the c parameter sampling. Smaller values
            increase accuracy but significantly increase the number of operations.
            If `c_precision` is halved, expect about 2 times more c steps per s
            (also runtime and RAM roughly double).
        step : float
            Defines the threshold reduction on single color conditions.Here, it is
            used to preliminarily remove suboptimal color condition parameters.
        """
        self.dict_all = {}
        self.td = target_definition
        self.dataset = dataset
        self.colors = colors
        self.thresholded_quantity = thresholded_quantity
        self.maximized_quantity_min = maximized_quantity_min
        self.buffered_threshold = buffered_threshold
        self.s_samples = s_samples
        self.c_precision = c_precision
        self.step = step

    def _filter_useful_cuts(
        self,
        params,
        purity_vals,
        completeness_vals,
    ):
        """
        Keep only optimal color cuts.

        Keep only color cut parameters that are optimal for at least one
        threshold step in the iterative optimization process.
        A cut is kept if it maximizes the non-thresholded quantity at any
        threshold value T = 1 - n * step, with T > buffered_threshold.

        This aligns the logic used in the iterative optimization in CalibrateLinear.
        """
        params = np.asarray(params, dtype=np.float32)
        purity_vals = np.asarray(purity_vals, dtype=np.float32)
        completeness_vals = np.asarray(completeness_vals, dtype=np.float32)

        if params.ndim == 1:
            params = params[:, None]  # ensure 2D

        if self.thresholded_quantity == "purity":
            Q_thr = purity_vals
            Q_max = completeness_vals
        else:
            Q_thr = completeness_vals
            Q_max = purity_vals

        # Sort by thresholded quantity (descending)
        order = np.argsort(Q_thr)[::-1]
        Q_thr = Q_thr[order]
        Q_max = Q_max[order]
        params = params[order]
        purity_vals = purity_vals[order]
        completeness_vals = completeness_vals[order]

        useful = np.zeros(len(Q_thr), dtype=bool)

        current_thresh = 1.0
        idx = 0
        best_Qmax_so_far = -np.inf

        while current_thresh > self.buffered_threshold and idx < len(Q_thr):

            if current_thresh > max(Q_thr):
                current_thresh -= self.step
                continue  # apparently sub-optimal, but essential for consistency with CalibrateLinear

            while idx < len(Q_thr) and Q_thr[idx] >= current_thresh:
                if Q_max[idx] > best_Qmax_so_far:
                    best_Qmax_so_far = Q_max[idx]
                    useful[idx] = True
                idx += 1
            current_thresh -= self.step

        return (
            params[useful],
            purity_vals[useful],
            completeness_vals[useful],
        )

    def _apply_cuts(self, z_lim, mag1, mag2, cross_colors=[], cpu_number=None):
        """
        Apply linear color–color cuts in parallel across a dataset.

        This method executes selection cuts in the a color–color plane
        including (mag1, mag2) as one of the dimensions, optionally
        distributing the computation across multiple CPU cores for efficiency.

        Parameters
        ----------
        z_lim : float
            Limiting redshift points.
        mag1 : array-like of float
            First magnitude or color feature used in the cuts.
        mag2 : array-like of float
            Second magnitude or color feature used in the cuts.
        cross_colors : list, optional
            List of additional colors (beyond mag1–mag2) for which to
            evaluate cuts combined with (mag1, mag2). Default is an
            empty list (only mag1–mag2 is considered).
        cpu_number : int or None, optional
            Number of CPUs to use for parallel processing.
            - If `None`, defaults to using `max(mp.cpu_count() - 2, 1)`.
            - If provided, uses `min(cpu_number, mp.cpu_count() - 2)` to avoid
              over-subscribing system resources.

        Returns
        -------
        None
            This method updates the internal class attribute `self.dict_all`
            and does not return any value.
        """
        N_cpu = (
            min(cpu_number, mp.cpu_count() - 2)
            if cpu_number
            else max(mp.cpu_count() - 2, 1)
        )
        N_cpu = max(1, N_cpu)  # Ensure at least 1 process

        # Precompute base arrays
        color1 = self.dataset[mag1].values.astype(np.float32) - self.dataset[
            mag2
        ].values.astype(np.float32)

        vmin, vmax = c_domain(color1)
        n_steps = int((vmax - vmin) / self.c_precision) + 1
        test_colors = np.linspace(vmin, vmax, n_steps, dtype=np.float32)

        # Create shared arrays
        shared_color1 = mp.Array("f", color1)  # 'f' for float32

        cross_color_arrays = {}
        cross_shared_arrays = {}
        if cross_colors:
            for mag3, mag4 in cross_colors:
                color2 = self.dataset[mag3].values.astype(np.float32) - self.dataset[
                    mag4
                ].values.astype(np.float32)
                cross_color_arrays[(mag3, mag4)] = color2
                cross_shared_arrays[(mag3, mag4)] = mp.Array("f", color2)

        # Proceed with the measurements
        shared_target_mask = mp.Array("b", [bool(x) for x in self.target_mask])

        if z_lim not in self.dict_all:
            self.dict_all[z_lim] = {}

        if (mag1, mag2) not in self.dict_all[z_lim]:
            self.dict_all[z_lim][(mag1, mag2)] = {}

        # One-color cuts
        with mp.Pool(
            N_cpu,
            initializer=init_worker,
            initargs=(shared_color1, None, shared_target_mask),
        ) as pool:
            results = pool.map(compute_for_simple_worker, test_colors)

        results_up = [res[:4] for res in results]
        results_low = [res[4:] for res in results]

        tp_up, fp_up, fn_up, tn_up = map(list, zip(*results_up))
        tp_low, fp_low, fn_low, tn_low = map(list, zip(*results_low))

        tp_up = np.asarray(tp_up, dtype=np.float32)
        fp_up = np.asarray(fp_up, dtype=np.float32)
        fn_up = np.asarray(fn_up, dtype=np.float32)
        tn_up = np.asarray(tn_up, dtype=np.float32)
        tp_low = np.asarray(tp_low, dtype=np.float32)
        fp_low = np.asarray(fp_low, dtype=np.float32)
        fn_low = np.asarray(fn_low, dtype=np.float32)
        tn_low = np.asarray(tn_low, dtype=np.float32)

        # Mask
        purity_up = tp_up / np.maximum(tp_up + fp_up, 1e-12)
        purity_low = tp_low / np.maximum(tp_low + fp_low, 1e-12)
        compl_up = tp_up / np.maximum(tp_up + fn_up, 1e-12)
        compl_low = tp_low / np.maximum(tp_low + fn_low, 1e-12)
        failure_up = fp_up / np.maximum(fp_up + tn_up, 1e-12)
        failure_low = fp_low / np.maximum(fp_low + tn_low, 1e-12)

        if self.thresholded_quantity == "purity":
            mask_threshold_up = purity_up > self.buffered_threshold
            mask_threshold_low = purity_low > self.buffered_threshold
            mask_maximized_up = compl_up > self.maximized_quantity_min
            mask_maximized_low = compl_low > self.maximized_quantity_min
        else:
            mask_threshold_up = compl_up > self.buffered_threshold
            mask_threshold_low = compl_low > self.buffered_threshold
            mask_maximized_up = purity_up > self.maximized_quantity_min
            mask_maximized_low = purity_low > self.maximized_quantity_min

        mask_up = (
            (mask_threshold_up)
            & (mask_maximized_up)
            & (compl_up > 0.0)
            & (purity_up > 0.0)
            & (failure_up < 1.0)
        )
        mask_low = (
            (mask_threshold_low)
            & (mask_maximized_low)
            & (compl_low > 0.0)
            & (purity_low > 0.0)
            & (failure_low < 1.0)
        )

        sub_dict = {}

        params_f, p_f, cpl_f = self._filter_useful_cuts(
            test_colors[mask_up],
            purity_up[mask_up],
            compl_up[mask_up],
        )
        sub_dict[0] = {
            "c": params_f[:, 0].tolist(),
            "purity": p_f.tolist(),
            "completeness": cpl_f.tolist(),
        }

        params_f, p_f, cpl_f = self._filter_useful_cuts(
            test_colors[mask_low],
            purity_low[mask_low],
            compl_low[mask_low],
        )
        sub_dict[1] = {
            "c": params_f[:, 0].tolist(),
            "purity": p_f.tolist(),
            "completeness": cpl_f.tolist(),
        }

        self.dict_all[z_lim][(mag1, mag2)][(mag1, mag2)] = sub_dict

        # Two-color cuts
        if cross_colors:

            for mag3, mag4 in cross_colors:

                shared_color2 = cross_shared_arrays[(mag3, mag4)]
                color2 = cross_color_arrays[(mag3, mag4)]

                sc_grid = []
                for s in self.s_samples:
                    c_min, c_max = c_domain_given_s(color1, color2, s)
                    n_c = int((c_max - c_min) / self.c_precision) + 1
                    c_grid = np.linspace(c_min, c_max, n_c, dtype=np.float32)

                    for c in c_grid:
                        sc_grid.append((s, c))

                with mp.Pool(
                    N_cpu,
                    initializer=init_worker,
                    initargs=(shared_color1, shared_color2, shared_target_mask),
                ) as pool:
                    results = pool.map(compute_for_cross_worker, sc_grid)

                # Transpose result list for each metric
                tp_all, fp_all, fn_all, tn_all = [], [], [], []

                for res in results:  # each res has 4 entries: one per condition
                    tp_c, fp_c, fn_c, tn_c = zip(*res)
                    tp_all.append(list(tp_c))
                    fp_all.append(list(fp_c))
                    fn_all.append(list(fn_c))
                    tn_all.append(list(tn_c))

                tp_all = np.asarray(tp_all, dtype=np.float32)
                fp_all = np.asarray(fp_all, dtype=np.float32)
                fn_all = np.asarray(fn_all, dtype=np.float32)
                tn_all = np.asarray(tn_all, dtype=np.float32)

                sc_grid = np.asarray(sc_grid, dtype=np.float32)

                # Mask
                purity = tp_all / np.maximum(tp_all + fp_all, 1e-12)
                compl = tp_all / np.maximum(tp_all + fn_all, 1e-12)
                failure = fp_all / np.maximum(fp_all + tn_all, 1e-12)

                if self.thresholded_quantity == "purity":
                    mask_threshold = purity > self.buffered_threshold
                    mask_maximized = compl > self.maximized_quantity_min
                else:
                    mask_threshold = compl > self.buffered_threshold
                    mask_maximized = purity > self.maximized_quantity_min

                mask_per_condition = (
                    (mask_threshold)
                    & (mask_maximized)
                    & (compl > 0.0)
                    & (purity > 0.0)
                    & (failure < 1.0)
                )

                # Save structure per condition index (0, 1, 2, 3)
                sub_dict = {}

                for cond_index in range(4):
                    mask = mask_per_condition[:, cond_index]

                    if np.any(mask):
                        params_f, p_f, cpl_f = self._filter_useful_cuts(
                            sc_grid[mask][:, :2],
                            purity[mask, cond_index],
                            compl[mask, cond_index],
                        )

                        sub_dict[cond_index] = {
                            "s": params_f[:, 0].tolist(),
                            "c": params_f[:, 1].tolist(),
                            "purity": p_f.tolist(),
                            "completeness": cpl_f.tolist(),
                        }

                self.dict_all[z_lim][(mag1, mag2)][(mag3, mag4)] = sub_dict

    def process(self, z_lim, CDZs, PDZ_x, cpu_number=None):
        """
        Apply all color cuts and compute summary statistics.

        This method iterates over all unique color combinations, applying linear
        and cross cuts using `_apply_cuts`. Results are accumulated internally
        in `self.dict_all`.

        Parameters
        ----------
        z_lim : float
            Limiting redshift.
        CDZs : list of array-like, default=None
            Cumulative redshift probability functions (normalization IS REQUIRED) used
            in the metrics computation. If None, point-estimates are used
            (whose key is defined by the "redshift" key in the data set).
        PDZ_x : array-like
            Redshift values where CDZs[i] is computed.
        cpu_number : int or None, optional
            Number of CPUs to use for parallel processing when applying cuts.
            - If `None`, defaults to `max(mp.cpu_count() - 2, 1)`.
            - If provided, the number of CPUs is capped at `mp.cpu_count() - 2`
              to prevent oversubscription.

        Returns
        -------
        None
            This method updates the internal class attribute `self.dict_all`
            and does not return any value.
        """
        if z_lim not in self.td.z_lim:
            raise ValueError(
                "z_lim must be one of the elements of TargetDefinition's z_lim"
            )

        self.target_mask = self.td._get_target_mask(self.dataset, [z_lim], CDZs, PDZ_x)[0]

        for i in range(len(self.colors)):
            self._apply_cuts(
                z_lim,
                self.colors[i][0],
                self.colors[i][1],
                cross_colors=self.colors[i + 1 : len(self.colors)],
                cpu_number=cpu_number,
            )
