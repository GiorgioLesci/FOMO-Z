"""
Derive optimal color selections based on color-color grids.

This module selects the optimal color-color grids,
given any limit on redshift and on other galaxy properties.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import copy
import multiprocessing as mp
import os
import time

import numpy as np
import yaml
from scipy.stats import beta
from tqdm import tqdm
from astropy.io import fits
import pandas as pd

from .dataset import Dataset
from .selection_definition import SelectionDefinition
from .tools import (
    c_domain,
    compute_metrics,
    prob_target,
    get_color_color_mask,
    update_dict,
)


class CalibrationGrid:
    """Calibrate grid color selections."""

    def __init__(
        self,
        dataset: Dataset,
        target_definition: SelectionDefinition,
        thresholded_quantity_min,
        maximized_quantity="completeness",
        grid_precision=0.15,
        alpha=0.1,
        step=0.0001,
        do_bootstrap=True,
        n_bootstraps=1000,
    ):
        r"""
        Class calibrating optimal grid color selections.

        This class implements a general approach which identifies optimal selection regions
        in color-color spaces by directly exploring discretized grids. At each point of
        these color-color grids, the following probability to find a target
        object is computed:

        $P_T = \frac{N_T}{N_S}\,,$

        where $N_T$ is the number of target objects in the color-color bin, and
        $N_S$ is the total number of objects in the reference sample falling in the bin.

        For each limiting redshift value specified in `SelectionDefinition`,
        this method searches for the set of cuts that
        ($\delta\mathcal{P}_{\rm step}$: `step`,
        $\mathcal{P}_{\rm min}$: `thresholded_quantity_min`,
        $\mathcal{C}$: `maximized_quantity`):
        - Update $\mathcal{P}_{\rm iter}^n = 1 - n\,\delta\mathcal{P}_{\rm step}$,
          where $n$ is the iteration index, starting from 0.
        - Remove the color-color bins where $P_T < \mathcal{P}_{\rm iter}^n$.
        - Compute the global purity $\mathcal{P}$, assuming that all color-color
          bins yield a significant measurement of $P_T$. If $\mathcal{P}>\mathcal{P}_{\rm min}$,
          repeat from step 1.
        - Mask not significant bins, based on the value of `alpha`. Then, recompute
          the selection metrics.

        Parameters
        ----------
        dataset : Dataset
            A `Dataset` object containing the dataset and band information.
        target_definition : SelectionDefinition
            A `SelectionDefinition` object containing target selections (e.g. redshift thresholds).
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
        grid_precision : float, default=0.15
            Precision of the color-color grids. A smaller value yields a finer,
            more accurate grid at the cost of increased computation, RAM usage, and
            storage requirements.
        alpha : float, default=0.1
            Significance level used to discard statistically non-robust grid bins.
            For each bin, let k be the number of target objects (successes) out of
            n total objects (trials), and let p denote the true per-bin purity.
            Assuming a Binomial likelihood P(k | p, n) and a uniform prior on p
            over the interval [0, 1], the posterior distribution P(p | k, n)
            follows a Beta distribution.
            A bin is considered statistically significant only if the posterior
            probability that its true purity lies below the minimum acceptable
            purity p_min satisfies:

                P(p < p_min | k, n) < alpha.

            Therefore, alpha represents the maximum tolerated posterior probability
            that a bin's true purity is below the required threshold.
            The default value alpha = 0.1 corresponds to allowing at most a 10%
            posterior probability that the bin purity is below p_min.
            To disable this statistical filtering and retain all bins,
            set alpha = 1.
        step : float, default=0.0001
            Defines the threshold reduction on grid elements. Default is 0.0001.
        do_bootstrap : bool, default=True
            If True, derive uncertainties on C, P, and F through bootstrap.
        n_bootstraps : int, default=1000
            Number of bootstrap resamples to perform. It defines
            the number of resampling in error estimation.

        Attributes
        ----------
        dict : dict
            Container for storing results based on color selections.
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

        Notes
        -----
        Purity maximization is not implemented yet. `maximized_quantity == "completeness"`
        is the only accepted value.
        """
        self.dataset = dataset.dataset
        self.bands = dataset.bands
        self.colors = dataset.colors
        self.CDZs = dataset.CDZs
        self.PDZ_x = dataset.PDZ_x
        self.weights = np.array(self.dataset["weights"].values)
        self.color_spaces = dataset.color_spaces
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
            raise NotImplementedError("Purity maximization is not implemented yet.")
            self.thresholded_quantity = "completeness"

        self.grid_precision = float(grid_precision)

        if alpha < 0 or alpha > 1:
            raise ValueError(
                f"alpha must range between 0 and 1. The current value is {alpha}."
            )
        self.alpha = alpha

        self.step = step
        self.do_bootstrap = do_bootstrap
        self.n_bootstraps = n_bootstraps

        # Initialize other attributes
        self.dict_steps = {}
        self.P = np.zeros(len(self.td.z_lim))
        self.C = np.zeros(len(self.td.z_lim))
        self.F = np.zeros(len(self.td.z_lim))
        self.C_std = np.full(len(self.td.z_lim), np.nan)
        self.P_std = np.full(len(self.td.z_lim), np.nan)
        self.F_std = np.full(len(self.td.z_lim), np.nan)

        self.dict = {}
        self._update_dict_metadata(self.dict, self.dataset)
        self.dict = update_dict(
            self.dict,
            self.dataset,
            self.dict["metadata"],
        )

        # Create the target masks
        self.target_mask = self.td._get_target_mask(
            self.dataset, self.td.z_lim, self.CDZs, self.PDZ_x
        )

        dataset_unweighted = self.dataset.copy()
        dataset_unweighted["weights"] = np.ones(len(self.dataset))
        self.target_mask_unweighted = self.td._get_target_mask(
            dataset_unweighted, self.td.z_lim, None, None
        )

    def _update_dict_metadata(self, dictionary, dataset):
        r"""
        Update the input dictionary's metadata, given the input data set.

        Parameters
        ----------
        dictionary : dict
            Dictionary.
        dataset : pandas.DataFrame
            Input dataset containing photometric bands.
        """
        dictionary["metadata"] = {}
        dictionary["metadata"]["bands"] = self.bands
        dictionary["metadata"]["colors"] = self.colors
        dictionary["metadata"]["color_spaces"] = self.color_spaces
        dictionary["metadata"]["grid_precision"] = self.grid_precision
        dictionary["metadata"]["z_l"] = [float(z) for z in self.td.z_lim]
        dictionary["metadata"]["z_l_is_lower_limit"] = self.td.is_zlim_lower
        dictionary["metadata"][f"{self.thresholded_quantity}_min"] = [
            float(T) for T in self.thresholded_quantity_min
        ]
        dictionary["metadata"]["grid_precision"] = self.grid_precision

        dictionary["metadata"]["color_limits"] = {}
        for cs in self.color_spaces:
            value1 = dataset[cs[0][0]].values - dataset[cs[0][1]].values
            value2 = dataset[cs[1][0]].values - dataset[cs[1][1]].values
            color1_min, color1_max = c_domain(value1, q_low=0.0001, q_high=0.9999)
            color2_min, color2_max = c_domain(value2, q_low=0.0001, q_high=0.9999)

            # These are used to reconstruct the bin edges from the metadata
            dictionary["metadata"]["color_limits"][cs[0]] = [
                color1_min,
                color1_max,
            ]
            dictionary["metadata"]["color_limits"][cs[1]] = [
                color2_min,
                color2_max,
            ]

    def _process_for_one_z(self, args):
        r"""
        Calibrate the color cut selection on grids for a single redshift threshold `z_l`.

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
        """
        i_z, store_iterations, cpu_number = args

        # Loop over color-color spaces to pre-compute statistics on the grids
        dict_stats = {}  # temporary dict containing grid statistics and masks

        for cs in self.color_spaces:

            # Dictionary initializations
            self.dict[cs]["mask"][i_z] = {}

            dict_stats[cs] = {}
            dict_stats[cs]["mask"] = {}
            dict_stats[cs]["color1_index"] = self.dict[cs]["color1_index"]
            dict_stats[cs]["color2_index"] = self.dict[cs]["color2_index"]
            dict_stats[cs]["valid"] = self.dict[cs]["valid"]

            # Probability to get target galaxies
            prob, N_good, N_total = prob_target(
                self.dict["metadata"],
                cs,
                self.dataset,
                self.target_mask[i_z],
                self.weights,
            )
            dict_stats[cs]["prob"] = prob
            dict_stats[cs]["k"] = N_good
            dict_stats[cs]["n"] = N_total

        # Remove bins with prob below a fixed threshold iteratively
        prob_min_bin = 1.0  # starting bin-level probability
        thresholded_quantity, P, C, F = 0.0, 0.0, 0.0, 0.0

        while prob_min_bin > 0.0:

            # Mask definition
            for cs in self.color_spaces:
                mask = dict_stats[cs]["prob"] >= prob_min_bin
                dict_stats[cs]["mask"][i_z] = mask

            mask = self.get_mask(i_z, dictionary=dict_stats)

            # Total completeness and purity
            TP, FP, FN, TN = compute_metrics(mask, self.target_mask[i_z], self.weights)
            C = TP / (TP + FN) if TP + FN > 0 else 0.0
            P = TP / (TP + FP) if TP + FP > 0 else 0.0

            # Check convergence
            if self.thresholded_quantity == "purity":
                thresholded_quantity = P
            elif self.thresholded_quantity == "completeness":
                thresholded_quantity = C
            else:
                raise ValueError(
                    f"Not allowed value for thresholded_quantity ({self.thresholded_quantity})."
                )

            if thresholded_quantity >= self.thresholded_quantity_min[i_z]:
                self.P[i_z] = P
                self.C[i_z] = C
                self.F[i_z] = FP / (FP + TN) if FP + TN > 0 else 0.0
                for cs in self.color_spaces:
                    self.dict[cs]["mask"][i_z] = dict_stats[cs]["mask"][i_z]
            else:
                break

            # Increase the bin-level threshold
            prob_min_bin -= self.step

        if prob_min_bin <= 0:
            errMsg = f"For z_l={self.td.z_lim[i_z]}, the convergence is attained for a value larger than the user-defined thresholded_quantity_min."
            errMsg += f" The current global P, C, and F are {self.P[i_z]}, {self.C[i_z]}, and {self.F[i_z]}, respectively."
            raise ValueError(errMsg)

        # Remove not robust bins, namely those with probability to have per-bin purity
        # p < prob_min_bin larger than alpha. Following the Bayes' theorem, we have
        # P(p | k, n) = P(k | p, n) P(p | n) / P(k | n),
        # where p is the per-bin purity, k is the number of successes, and n is the
        # number of trials. We assume that P(k | p, n) is a Binomial distribution and
        # P(p | n) = [0, 1] is a uniform prior.
        # Our selection criterion is P(p < p_min | k, n) < alpha.
        dict_significant = {}

        for cs in self.color_spaces:

            dict_significant[cs] = {}
            dict_significant[cs]["mask"] = {}
            shape = dict_stats[cs]["prob"].shape
            dict_significant[cs]["mask"] = np.zeros(shape, dtype=bool)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    k = dict_stats[cs]["k"][i][j]
                    n = dict_stats[cs]["n"][i][j]
                    if n == 0:
                        continue
                    # Given the observed data (k successes out of n), compute the posterior
                    # probability that (true purity < prob_min_bin) is smaller than alpha.
                    significant = (
                        beta.cdf(prob_min_bin, k + 1, np.maximum(n - k, 0) + 1)
                        < self.alpha
                    )
                    dict_significant[cs]["mask"][i, j] = significant

        # Re-define the masks and re-evaluate the metrics
        for cs in self.color_spaces:
            self.dict[cs]["mask"][i_z] = (
                dict_stats[cs]["mask"][i_z] & dict_significant[cs]["mask"]
            )

        mask = self.get_mask(i_z)
        TP, FP, FN, TN = compute_metrics(mask, self.target_mask[i_z], self.weights)
        self.P[i_z] = TP / (TP + FP) if TP + FP > 0 else 0.0
        self.C[i_z] = TP / (TP + FN) if TP + FN > 0 else 0.0
        self.F[i_z] = FP / (FP + TN) if FP + TN > 0 else 0.0

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

            # Bootstrap sample
            indices = np.random.choice(
                len(self.dataset), size=len(self.dataset), replace=True
            )
            bt_sample = self.dataset.iloc[indices].copy()

            # Pre-compute colors
            dict_temp = {}
            dict_temp = update_dict(
                dict_temp,
                bt_sample,
                self.dict["metadata"],
            )

            # Set the color-color masks equal to the calibrated ones
            for cs in self.color_spaces:
                dict_temp[cs]["mask"] = {}
                for i_z in range(len(self.td.z_lim)):
                    dict_temp[cs]["mask"][i_z] = self.dict[cs]["mask"][i_z]

            # Compute the metrics
            C_temp = np.zeros(len(self.td.z_lim))
            P_temp = np.zeros(len(self.td.z_lim))
            F_temp = np.zeros(len(self.td.z_lim))

            bt_CDZs = self.CDZs[indices] if self.CDZs is not None else None
            target_mask = self.td._get_target_mask(
                bt_sample, self.td.z_lim, bt_CDZs, self.PDZ_x
            )
            for i_z in range(len(self.td.z_lim)):
                mask = self.get_mask(i_z, dictionary=dict_temp)
                TP, FP, FN, TN = compute_metrics(
                    mask, target_mask[i_z], bt_sample["weights"]
                )
                C_temp[i_z] = TP / (TP + FN) if TP + FN > 0 else 0.0
                P_temp[i_z] = TP / (TP + FP) if TP + FP > 0 else 0.0
                F_temp[i_z] = FP / (FP + TN) if FP + TN > 0 else 0.0

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
        Perform the iterative search for optimal color selection grids.

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
        print("\nRunning the grid color selection calibration...")

        for i_z in tqdm(range(len(self.td.z_lim)), desc="Looping over z_lim points"):
            self._process_for_one_z((i_z, store_iterations, cpu_number))

        # Compute bootstrap errors
        if self.do_bootstrap:
            self._estimate_errors()

        print(f"Converged!")
        print(f"P_tot = {self.P} +- {self.P_std}")
        print(f"C_tot = {self.C} +- {self.C_std}")
        print(f"F_tot = {self.F} +- {self.F_std}")

        print("Grid calibration complete.")

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
        Save the optimal color-color grid calibration to a FITS file.

        The file contains one extension group per color-color space:
        - An ImageHDU with the 3D boolean mask (shape: n_zlim x n_color1bins x n_color2bins).
        - A BinTableHDU with per-object arrays (valid, color1_index, color2_index)
          and bin edges (color1_edges, color2_edges).

        Global metadata (z_l, z_l_is_lower_limit, etc.) are stored in the
        primary HDU header.

        Parameters
        ----------
        output_dir : str
            Path to the directory where the file will be saved.
        file_name : str
            Base name of the output file (without extension).
            The output file is named ``FOMO-Z_{file_name}_grid.fits``.
        reference_sample_name : str, optional
            Name of the reference sample, stored in the FITS header.
        notes : str, optional
            Additional notes stored in the FITS header.
        write_metrics : bool, optional
            If True, write the `FOMO-Z_{file_name}_metrics.csv` file containing
            purity, completeness, and failure rate
            as a function of limiting redshift, and their errors.
        write_redshift_distributions : bool, optional
            If True, write the `FOMO-Z_{file_name}_redshift_distributions.csv` file,
            containing the binned redshift distributions of color-selected objects.
            This file contains the redshift bins and the histograms for each limiting
            redshift point provided to `SelectionDefinition`.
            Histograms are derived using the `get_redshift_distribution` method.
            The first two columns of the file contain bin edges and bin centers,
            respectively. Subsequent columns contain the redshift distributions
            corresponding to each limiting redshift value provided to `SelectionDefinition`.
            These histograms are normalized such that their integral is 1. All histograms
            share the same bin edges. If weights were provided to `Dataset`, these histograms
            are weighted accordingly.
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

        # --- Primary HDU: global metadata ---
        primary_header = fits.Header()
        primary_header["FOMOVER"] = ("FOMO-Z", "Package name")
        primary_header["REFSAMP"] = (reference_sample_name, "Reference sample name")
        primary_header["NOTES"] = ("Produced with the FOMO-Z code. " + notes)[:70]

        meta = self.dict["metadata"]
        primary_header["BANDS"] = " ".join(meta["bands"])

        # z_l array: stored as a space-separated string in header, or as a small table.
        # We use a dedicated BinTableHDU for z_l to avoid header length limits.
        primary_header["ZLISLOW"] = (
            int(meta["z_l_is_lower_limit"]),
            "1 if z_l values are lower limits",
        )
        primary_header["NZL"] = (
            len(meta["z_l"]),
            "Number of redshift threshold values",
        )

        # Identify color-color space keys (all keys except "metadata")
        color_spaces = [k for k in self.dict if k != "metadata"]
        primary_header["NCS"] = (len(color_spaces), "Number of color-color spaces")

        primary_hdu = fits.PrimaryHDU(header=primary_header)
        hdul = fits.HDUList([primary_hdu])

        # --- Extension 1: z_l values ---
        z_l_col = fits.Column(
            name="z_l",
            format="D",  # double precision float
            array=np.asarray(meta["z_l"], dtype=np.float64),
        )
        z_l_hdu = fits.BinTableHDU.from_columns([z_l_col])
        z_l_hdu.name = "ZL_VALUES"
        hdul.append(z_l_hdu)

        # --- One ImageHDU + one BinTableHDU per color-color space ---
        for cs in color_spaces:
            cs_data = self.dict[cs]

            # cs is a tuple like ((band1, band2), (band3, band4))
            # Encode it as a header string: "band1-band2_band3-band4"
            cs_str = f"{cs[0][0]}-{cs[0][1]}_{cs[1][0]}-{cs[1][1]}"

            # 3D boolean mask: shape (n_zlim, n_color1bins, n_color2bins)
            mask_array = np.stack(
                [cs_data["mask"][i] for i in range(len(cs_data["mask"]))],
                axis=0,
            ).astype(
                np.uint8
            )  # shape: (n_zlim, n_color1bins, n_color2bins)

            mask_hdu = fits.ImageHDU(data=mask_array)
            mask_hdu.name = f"MASK_{cs_str}"[:68]  # FITS extension name limit
            mask_hdu.header["CSNAME"] = (cs_str, "Color-color space identifier")
            mask_hdu.header["C1BAND1"] = (cs[0][0], "Color 1: band 1")
            mask_hdu.header["C1BAND2"] = (cs[0][1], "Color 1: band 2")
            mask_hdu.header["C2BAND1"] = (cs[1][0], "Color 2: band 1")
            mask_hdu.header["C2BAND2"] = (cs[1][1], "Color 2: band 2")
            hdul.append(mask_hdu)

            # Per-object arrays in a BinTableHDU
            valid = np.asarray(cs_data["valid"], dtype=bool)
            color1_index = np.asarray(cs_data["color1_index"], dtype=np.int32)
            color2_index = np.asarray(cs_data["color2_index"], dtype=np.int32)

            cols = fits.ColDefs(
                [
                    fits.Column(name="valid", format="L", array=valid),
                    fits.Column(name="color1_index", format="J", array=color1_index),
                    fits.Column(name="color2_index", format="J", array=color2_index),
                ]
            )

            table_hdu = fits.BinTableHDU.from_columns(cols)
            table_hdu.name = f"OBJ_{cs_str}"[:68]
            hdul.append(table_hdu)

        # Write
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"FOMO-Z_{file_name}.fits")
        hdul.writeto(file_path, overwrite=True)

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
        Get the color-color space keys.

        Returns the list of color-color space keys contained in `self.dict`.

        Returns
        ------
        keys : list of str
            List of color-color space keys.
        """
        return [key for key in list(self.dict.keys()) if key != "metadata"]

    def get_mask(self, zl_index, color_space="all", dictionary=None):
        """
        Get the calibration mask.

        Returns the calibration mask relative to the input limiting redshift.

        Parameters
        ------
        zl_index : int
            Index of the redshift threshold in SelectionDefinition.z_lim.
        color_space : str or tuple, optional
            It can be either "all" (all color-color spaces) or one of
            the available color-color spaces, in the form `((band1, band2), (band3, band4))`.
            Available color-color spaces can be obtained from `self.get_keys()`.
        dictionary : dict, optional
            The color-color space dictionary, containing grid masks. If not provided,
            the internal dictionary (`self.dict`) is used.

        Returns
        -------
        mask : numpy.array
            Array of booleans.
        """
        if color_space == "all":
            color_space = self.get_keys()
        else:
            if color_space not in self.get_keys():
                raise ValueError(f"Not allowed color_space value {color_space}.")
            color_space = [color_space]

        if dictionary is None:
            dictionary = self.dict

        return get_color_color_mask(zl_index, color_space, dictionary)

    def get_redshift_distribution(self, zl_index, bin_edges, normalize=True):
        """
        Get the redshift distribution of color-selected objects.

        Returns the redshift distribution relative to the input limiting redshift.
        If weights were provided to `Dataset`, these histograms are weighted accordingly.

        Parameters
        ------
        zl_index : int
            Index of the redshift threshold in SelectionDefinition.z_lim.
        bin_edges : array-like
            Number of histogram bins.
        normalize : bool, optional
            If true, the output histogram is a PDF. Otherwise, it is a count histogram.

        Returns
        -------
        histogram : numpy.array
            Array of histogram values.
        """
        mask = self.get_mask(zl_index, color_space="all")
        dataset = self.dataset["redshift"][mask]
        hist, _ = np.histogram(
            dataset, bins=bin_edges, density=normalize, weights=self.weights[mask]
        )
        return hist
