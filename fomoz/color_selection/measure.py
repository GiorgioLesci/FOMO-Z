"""
MEASURE COLOR CUT PERFORMANCE.

This module provides the computation of color condition metrics,
given any lower limit in redshift and condition parameter values.

:Author: Giorgio Francesco Lesci <giorgio.lesci2 [at] unibo [dot] it>
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from itertools import combinations

from .auxiliary import compute_for_simple, compute_for_cross


class Measure:
    """
    Class defining color selection completeness, purity, and
    foreground failure rate for all color conditions, given any
    lower redshift limit values and color condition parameters.

    This class standardizes input photometric data, handles missing values,
    constructs color arrays from provided bands, and computes color selection
    metrics. These selections are defined as linear relations between
    colors and are evaluated over a grid of slope–intercept combinations.

    Attributes
    ----------
    dataset : pandas.DataFrame
        The input dataset with standardized band magnitudes and redshift column.
    bands : list of str
        List of photometric band names to be used.
    colors : list of tuple
        All unique 2-band combinations generated from `bands`, used to form
        color indices (e.g., g-r, r-i).
    z_lim : numpy.ndarray of float32
        Array of redshift lower limits.
    sc_grid : list of tuple
        Grid of slope–intercept pairs (s, c) used for defining linear
        color–color cuts.
    dict_all : dict
        Dictionary for storing computed metrics across color cuts.

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
    c_samples : array-like, default=np.linspace(-5, 5, 100)
        Array of intercept values for generating linear cuts in color–color
        space.
    """

    def __init__(
        self,
        dataset,
        bands,
        z_lim,
        z_gal_string="redshift",
        missing_data=-99,
        s_samples=np.linspace(0.3, 1.7, 30),
        c_samples=np.linspace(-5, 5, 100),
    ):
        """ """
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

        for band in bands:
            dataset[band] = dataset[band].replace(missing_data, np.nan)
        dataset = dataset.rename(columns={z_gal_string: "redshift"})
        dataset["redshift"] = dataset["redshift"].replace(missing_data, np.nan)

        self.dataset = dataset
        self.bands = bands
        self.colors = list(combinations(self.bands, 2))

        if isinstance(z_lim, (float, int)):
            self.z_lim = np.array([float(z_lim)])
        else:
            self.z_lim = z_lim
        for z_l in self.z_lim:
            if z_l < 0:
                raise ValueError("z_lim values cannot be negative")

        s_samples = s_samples.astype(np.float32)
        c_samples = c_samples.astype(np.float32)
        self.sc_grid = [(s, c) for s in s_samples for c in c_samples]
        self.dict_all = {}

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

        # Precompute base arrays
        color = self.dataset[mag1].values.astype(np.float32) - self.dataset[
            mag2
        ].values.astype(np.float32)
        redshift = self.dataset["redshift"].values.astype(np.float32)
        test_colors = np.linspace(color.min(), color.max(), 300, dtype=np.float32)[1:-1]

        # Create pool once
        with mp.Pool(N_cpu) as pool:

            for z_l in tqdm(
                self.z_lim,
                desc=f"Processing all z_l points for the color ({mag1}-{mag2})",
            ):

                if z_l not in self.dict_all:
                    self.dict_all[z_l] = {}

                if (mag1, mag2) not in self.dict_all[z_l]:
                    self.dict_all[z_l][(mag1, mag2)] = {}

                mask_good = redshift > z_l
                mask_bad = redshift < z_l

                # One-color cuts
                color_cut_args = [(tc, color, redshift, z_l) for tc in test_colors]
                results = pool.map(compute_for_simple, color_cut_args)

                results_up = [res[:4] for res in results]
                results_low = [res[4:] for res in results]

                tp_up, fp_up, fn_up, tn_up = map(list, zip(*results_up))
                tp_low, fp_low, fn_low, tn_low = map(list, zip(*results_low))

                self.dict_all[z_l][(mag1, mag2)][(mag1, mag2)] = {
                    "c": [np.float32(tc) for tc in test_colors],
                    "TP": list(zip(tp_up, tp_low)),
                    "FP": list(zip(fp_up, fp_low)),
                    "FN": list(zip(fn_up, fn_low)),
                    "TN": list(zip(tn_up, tn_low)),
                }

                # Two-color cuts
                if cross_colors:

                    for mag3, mag4 in cross_colors:
                        color1 = self.dataset[mag1].values.astype(
                            np.float32
                        ) - self.dataset[mag2].values.astype(np.float32)
                        color2 = self.dataset[mag3].values.astype(
                            np.float32
                        ) - self.dataset[mag4].values.astype(np.float32)

                        sc_args = [
                            (s, c, color1, color2, redshift, z_l)
                            for s, c in self.sc_grid
                        ]

                        results = pool.map(compute_for_cross, sc_args)

                        # Transpose result list for each metric
                        tp_all, fp_all, fn_all, tn_all = [], [], [], []

                        for res in results:  # each res has 4 entries: one per condition
                            tp_c, fp_c, fn_c, tn_c = zip(*res)
                            tp_all.append(list(tp_c))
                            fp_all.append(list(fp_c))
                            fn_all.append(list(fn_c))
                            tn_all.append(list(tn_c))

                        self.dict_all[z_l][(mag1, mag2)][(mag3, mag4)] = {
                            "s": [s for s, _ in self.sc_grid],
                            "c": [c for _, c in self.sc_grid],
                            "TP": tp_all,
                            "FP": fp_all,
                            "FN": fn_all,
                            "TN": tn_all,
                        }

    def process(self, cpu_number=None):
        """
        Apply all color–color cuts across the dataset and compute
        summary statistics.

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
