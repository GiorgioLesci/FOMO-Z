"""
MEASURE COLOR CUT PERFORMANCE.

This module provides the computation of color condition metrics,
given any lower limit in redshift and condition parameter values.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import multiprocessing as mp
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from .auxiliary import compute_for_cross, compute_for_simple, c_domain_given_s


# Globals visible to workers
GLOBAL_color1 = None
GLOBAL_color2 = None
GLOBAL_redshift = None


def init_worker(shared_c1, shared_c2, shared_redshift):
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
    shared_redshift : multiprocessing.sharedctypes.SynchronizedArray
        Shared memory array for galaxy redshift values.
    """
    global GLOBAL_color1, GLOBAL_color2, GLOBAL_redshift
    GLOBAL_color1 = np.frombuffer(shared_c1.get_obj(), dtype=np.float32) if shared_c1 else None
    GLOBAL_color2 = np.frombuffer(shared_c2.get_obj(), dtype=np.float32) if shared_c2 else None  
    GLOBAL_redshift = np.frombuffer(shared_redshift.get_obj(), dtype=np.float32)


def compute_for_simple_worker(args):
    """
    Wrap `compute_for_simple` using global data set by `init_worker`.

    Parameters are passed via `args` to remain compatible with `pool.map`.
    See `compute_for_simple` for full documentation.
    """
    global GLOBAL_color1, GLOBAL_redshift
    c, z_l = args
    return compute_for_simple((c, GLOBAL_color1, GLOBAL_redshift, z_l))


def compute_for_cross_worker(args):
    """
    Wrap `compute_for_cross` using global data set by `init_worker`.

    Parameters are passed via `args` to remain compatible with `pool.map`.
    See `compute_for_cross` for full documentation.
    """
    global GLOBAL_color1, GLOBAL_color2, GLOBAL_redshift
    s, c, z_l = args
    return compute_for_cross((s, c, GLOBAL_color1, GLOBAL_color2, GLOBAL_redshift, z_l))


class Measure:
    """Measure the metrics of color conditions."""

    def __init__(
        self,
        dataset,
        bands,
        z_lim,
        z_gal_string="redshift",
        missing_data=-99,
        s_samples=np.linspace(0.3, 1.7, 30),
        c_precision=0.05,
        purity_initial_threshold=0.9,
    ):
        """
        Measure the metrics of color conditions.

        Class defining color selection completeness, purity, and
        foreground failure rate for all color conditions, given any
        lower redshift limit values and color condition parameters.

        This class standardizes input photometric data, handles missing values,
        constructs color arrays from provided bands, and computes color selection
        metrics. These selections are defined as linear relations between
        colors and are evaluated over a grid of slope–intercept combinations.

        Parameters
        ----------
        dataset : pandas.DataFrame
            Input photometric dataset containing magnitude measurements and a
            redshift column.
        bands : list of str
            Names of the photometric bands to include in the analysis.
        z_lim : array-like
            Array of redshift limits/thresholds to be used in selection tests.
        z_gal_string : str, default='redshift'
            Column name in `dataset` that contains the galaxy redshift values.
        missing_data : int or float, default=-99
            Value used in the dataset to denote missing entries. These are replaced
            with NaN for consistency.
        s_samples : array-like, default=np.linspace(0.3, 1.7, 30)
            Array of slope values for generating linear cuts in color–color space.
            These samples must be strictly positive.
        c_precision : float, default=0.05
            Precision of the c parameter sampling. Smaller values increase accuracy
            but significantly increase the number of operations. If you halve `c_precision`,
            expect about 2 times more c steps per s (also runtime and RAM roughly double).
        purity_initial_threshold : float, default=0.9
            Initial threshold on purity. Color conditions with purity lower than
            this value are not stored. This is useful to avoid filling memory
            with unused information.
        """
        assert isinstance(
            dataset, pd.DataFrame
        ), "Input dataset must be a pandas DataFrame"

        if z_gal_string not in dataset.columns:
            errMsg = f"z_gal_string='{z_gal_string}' "
            errMsg += "not found in dataset columns."
            raise KeyError(errMsg)

        if "c" in bands or "s" in bands:
            raise ValueError("Magnitude keys cannot equal 'c' or 's'.")

        if z_gal_string in bands:
            errMsg = f"z_gal_string='{z_gal_string}' conflicts "
            errMsg += f"with a band name in {bands}."
            raise ValueError(errMsg)

        if "redshift" in dataset.columns and z_gal_string != "redshift":
            errMsg = "Dataset already contains a 'redshift' column. "
            errMsg += "Please rename it or set z_gal_string='redshift'."
            raise ValueError(errMsg)

        needed_columns = [z_gal_string] + bands
        self.dataset = dataset[needed_columns].copy()
        for band in bands:
            self.dataset[band] = self.dataset[band].replace(missing_data, np.nan)
        self.dataset = self.dataset.rename(columns={z_gal_string: "redshift"})
        self.dataset["redshift"] = self.dataset["redshift"].replace(missing_data, np.nan)

        self.bands = bands
        self.colors = list(combinations(self.bands, 2))

        if isinstance(z_lim, (float, int)):
            self.z_lim = np.array([float(z_lim)])
        elif isinstance(z_lim, list):
            self.z_lim = np.array(z_lim)
        else:
            self.z_lim = z_lim

        for z_l in self.z_lim:
            if z_l < 0:
                raise ValueError("z_lim values cannot be negative")

        if np.any(s_samples <= 0):
            raise ValueError(
                "All slope values in s_samples must be strictly positive. "
                f"Found non-positive values: {s_samples[s_samples <= 0]}"
            )
        self.s_samples = s_samples.astype(np.float32)
        self.c_precision = float(c_precision)
        self.dict_all = {}
        self.purity_initial_threshold = purity_initial_threshold

    def _apply_cuts(self, mag1, mag2, cross_colors=[], cpu_number=None):
        """
        Apply linear color–color cuts in parallel across a dataset.

        This method executes selection cuts in the a color–color plane
        including (mag1, mag2) as one of the dimensions, optionally
        distributing the computation across multiple CPU cores for efficiency.

        Parameters
        ----------
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
        redshift = self.dataset["redshift"].values.astype(np.float32)

        vmin, vmax = np.quantile(
            color1[np.isfinite(color1)], [0.01, 0.99]
        )
        n_steps = int((vmax - vmin) / self.c_precision) + 1
        test_colors = np.linspace(vmin, vmax, n_steps, dtype=np.float32)

        # Create shared arrays
        shared_color1 = mp.Array('f', color1)  # 'f' for float32
        shared_redshift = mp.Array('f', redshift)

        cross_color_arrays = {}
        cross_shared_arrays = {}
        if cross_colors:
            for mag3, mag4 in cross_colors:
                color2 = self.dataset[mag3].values.astype(np.float32) - self.dataset[mag4].values.astype(np.float32)
                cross_color_arrays[(mag3, mag4)] = color2
                cross_shared_arrays[(mag3, mag4)] = mp.Array('f', color2)

        # Loop over limiting redshift points
        for z_l in tqdm(
            self.z_lim,
            desc=f"Processing the color-color spaces that include ({mag1}-{mag2})",
        ):

            if z_l not in self.dict_all:
                self.dict_all[z_l] = {}

            if (mag1, mag2) not in self.dict_all[z_l]:
                self.dict_all[z_l][(mag1, mag2)] = {}

            mask_good = redshift > z_l
            mask_bad = redshift < z_l

            # One-color cuts
            color_cut_args = [(tc, z_l) for tc in test_colors]
            with mp.Pool(
                N_cpu,
                initializer=init_worker,
                initargs=(shared_color1, None, shared_redshift),
            ) as pool:
                results = pool.map(compute_for_simple_worker, color_cut_args)

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
            f_low_up = fp_up / np.maximum(fp_up + tn_up, 1e-12)
            f_low_low = fp_low / np.maximum(fp_low + tn_low, 1e-12)
            mask_up = (
                (purity_up > self.purity_initial_threshold)
                & (compl_up > 0.0)
                & (f_low_up < 1.0)
            )
            mask_low = (
                (purity_low > self.purity_initial_threshold)
                & (compl_low > 0.0)
                & (f_low_low < 1.0)
            )

            sub_dict = {}
            sub_dict[0] = {
                "c": test_colors[mask_up].tolist(),
                "purity": purity_up[mask_up].astype(np.float32).tolist(),
                "completeness": compl_up[mask_up].astype(np.float32).tolist(),
            }
            sub_dict[1] = {
                "c": test_colors[mask_low].astype(np.float32).tolist(),
                "purity": purity_low[mask_low].astype(np.float32).tolist(),
                "completeness": compl_low[mask_low].astype(np.float32).tolist(),
            }

            self.dict_all[z_l][(mag1, mag2)][(mag1, mag2)] = sub_dict

            # Two-color cuts
            if cross_colors:

                for mag3, mag4 in cross_colors:

                    shared_color2 = cross_shared_arrays[(mag3, mag4)]
                    color2 = cross_color_arrays[(mag3, mag4)]

                    sc_args = []
                    sc_grid = []
                    for s in self.s_samples:
                        c_min, c_max = c_domain_given_s(color1, color2, s)
                        n_c = int((c_max - c_min) / self.c_precision) + 1
                        c_grid = np.linspace(c_min, c_max, n_c, dtype=np.float32)

                        for c in c_grid:
                            sc_args.append((s, c, z_l))
                            sc_grid.append((s, c))

                    with mp.Pool(
                        N_cpu,
                        initializer=init_worker,
                        initargs=(shared_color1, shared_color2, shared_redshift),
                    ) as pool:
                        results = pool.map(compute_for_cross_worker, sc_args)

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
                    f_low = fp_all / np.maximum(fp_all + tn_all, 1e-12)
                    mask_per_condition = (
                        (purity > self.purity_initial_threshold)
                        & (compl > 0.0)
                        & (f_low < 1.0)
                    )

                    # Save structure per condition index (0, 1, 2, 3)
                    sub_dict = {}

                    for cond_index in range(4):
                        mask = mask_per_condition[:, cond_index]

                        if np.any(mask):
                            sub_dict[cond_index] = {
                                "s": sc_grid[mask, 0].tolist(),
                                "c": sc_grid[mask, 1].tolist(),
                                "purity": purity[mask, cond_index]
                                .astype(np.float32)
                                .tolist(),
                                "completeness": compl[mask, cond_index]
                                .astype(np.float32)
                                .tolist(),
                            }

                    self.dict_all[z_l][(mag1, mag2)][(mag3, mag4)] = sub_dict

    def process(self, cpu_number=None):
        """
        Apply all color cuts and compute summary statistics.

        This method iterates over all unique color combinations, applying linear
        and cross cuts using `_apply_cuts`. Results are accumulated internally
        in `self.dict_all`.

        Parameters
        ----------
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
        print("Computing color cuts given the following set of bands:", self.bands)

        for i in range(len(self.colors)):
            self._apply_cuts(
                self.colors[i][0],
                self.colors[i][1],
                cross_colors=self.colors[i + 1 : len(self.colors)],
                cpu_number=cpu_number,
            )
