"""
Derive optimal color selections based on color-color grids.

This module selects the optimal color-color grids,
given any condition on redshift or any other galaxy properties.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import copy
import os
import warnings

import numpy as np
from scipy.stats import beta
from tqdm import tqdm
from astropy.io import fits
import pandas as pd

from ..data.dataset import Dataset
from ..data.selection_definition import SelectionDefinition
from ..auxiliary import (
    compute_metrics,
    prob_target,
    get_color_color_mask,
    update_dict_color_limits,
    update_dict_metadata,
    update_dict,
)
from ..auxiliary.distributions import get_distribution as compute_distribution


class GridClassification:
    """Calibrate grid color selections."""

    def __init__(
        self,
        dataset: Dataset,
        selection_definition: SelectionDefinition,
        purity_min,
        grid_precision=0.15,
        alpha=1,
        step=0.001,
        do_bootstrap=True,
        n_bootstraps=1000,
        verbose=True,
    ):
        r"""
        Calibrate optimal grid-based color selections.

        This class identifies selection regions in color-color spaces by
        exploring discretized grids. In each color-color bin, the probability
        of finding a target object is estimated as

        .. math::

            P_{\rm T} = \frac{N_{\rm T}}{N_{\rm S}},

        where :math:`N_{\rm T}` is the number of target objects in the bin
        and :math:`N_{\rm S}` is the total number of reference objects in
        the same bin.

        For each limiting redshift specified in ``selection_definition``,
        let :math:`\delta\mathcal{P}_{\rm step}` denote ``step`` and
        :math:`\mathcal{P}_{\rm min}` denote ``purity_min``.
        The optimization proceeds as follows:

        1. Progressively lower the bin purity threshold from unity
           in increments of :math:`\delta\mathcal{P}_{\rm step}`.
           After :math:`n` decrements, the threshold is

           .. math::

               \mathcal{P}_{\rm iter}^{n}
               = 1 - n\,\delta\mathcal{P}_{\rm step}.

        2. Retain bins satisfying
           :math:`P_{\rm T}\geq\mathcal{P}_{\rm iter}^{n}`.

        3. Evaluate the global purity :math:`\mathcal{P}` without applying
           statistical filtering to individual bins. Continue relaxing
           the threshold while the combined selection satisfies
           :math:`\mathcal{P}>\mathcal{P}_{\rm min}`.

        4. Apply the statistical filtering controlled by ``alpha`` to the
           retained bins, then recompute the selection metrics.

        Bootstrap uncertainties on the resulting metrics can also be estimated.

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
        grid_precision : float, default=0.15
            Bin width of the color-color grids. Smaller values provide finer
            sampling at greater computational cost, memory usage, and
            storage requirements.
        alpha : float, default=1
            Maximum tolerated posterior probability that a bin's true
            purity lies below the minimum acceptable purity.

            For a bin containing :math:`k` target objects out of
            :math:`n` total objects, assume a binomial likelihood and a
            uniform prior on the true purity :math:`p` over :math:`[0,1]`.
            The posterior distribution is then

            .. math::

                p \mid k,n \sim \operatorname{Beta}(k+1,n-k+1).

            A bin is retained only if

            .. math::

                \Pr(p < p_{\rm min} \mid k,n) < \alpha,

            where :math:`p_{\rm min}` is the minimum acceptable bin purity.
            For example, ``alpha=0.1`` requires the posterior probability
            of falling below this threshold to be less than 10%.
            Set ``alpha=1`` to disable statistical filtering.
        step : float, default=0.001
            Decrement of the bin purity threshold at each optimization
            iteration, :math:`\delta\mathcal{P}_{\rm step}`.
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
            Calibrated grid-based color selections and associated results.
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
        self.color_spaces = dataset.color_spaces
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

        self.dict = update_dict_metadata(
            self.dict,
            self.bands,
            self.colors,
            self.color_spaces,
            self.grid_precision,
            self.td.z_lim,
            self.td.is_zlim_lower,
        )
        self.dict = update_dict_color_limits(self.dict, self.dataset)

        self.dict = update_dict(
            self.dict,
            self.dataset,
            self.dict["metadata"],
        )
        self.dict["metadata"][f"{self.thresholded_quantity}_min"] = [
            float(T) for T in self.thresholded_quantity_min
        ]

        # Create the target masks
        self.target_mask = self.td._get_target_mask(
            self.dataset,
            self.td.z_lim,
            self.CDZs,
            self.PDZ_x,
            self.z_gal_string,
            self.missing_data,
        )

        dataset_unweighted = self.dataset.copy()
        dataset_unweighted["weights"] = np.ones(len(self.dataset))
        self.target_mask_unweighted = self.td._get_target_mask(
            dataset_unweighted,
            self.td.z_lim,
            None,
            None,
            self.z_gal_string,
            self.missing_data,
        )

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
            errMsg = f"The convergence is attained for a value larger than the user-defined thresholded_quantity_min."
            errMsg += f" The current global P, C, and F are {self.P[i_z]}, {self.C[i_z]}, and {self.F[i_z]}, respectively."
            error = ValueError(errMsg)
            error.calibration_purity = float(self.P[i_z])
            raise error

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

        for i in tqdm(
            range(self.n_bootstraps),
            desc="Deriving bootstrap errors",
            disable=not self.verbose,
        ):

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
                bt_sample,
                self.td.z_lim,
                bt_CDZs,
                self.PDZ_x,
                self.z_gal_string,
                self.missing_data,
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

    def process(self):
        r"""
        Perform the iterative search for optimal color selection grids.

        Here, either completeness or purity are maximized, depending on what was
        set in the `SelectionDefinition` instance given in input to the constructor of this class.

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
            print("\nRunning the grid color selection calibration...")

        cpu_number = None  # to be implemented
        store_iterations = False  # to be implemented
        for i_z in tqdm(
            range(len(self.td.z_lim)),
            desc="Looping over z_lim points",
            disable=not self.verbose,
        ):
            self._process_for_one_z((i_z, store_iterations, cpu_number))

        # Compute bootstrap errors
        if self.do_bootstrap:
            self._estimate_errors()

        if self.verbose:
            print("Grid calibration complete.")

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

        # --- Primary HDU: global metadata ---
        primary_header = fits.Header()
        primary_header["FOMOVER"] = ("FOMO-Z", "Package name")
        primary_header["REFSAMP"] = (reference_sample_name, "Reference sample name")
        primary_header["NOTES"] = ("Produced with the FOMO-Z code. " + notes)[:70]

        meta = self.dict["metadata"]
        primary_header["BANDS"] = " ".join(meta["bands"])
        primary_header["ZLISLOW"] = (
            int(meta["z_l_is_lower_limit"]),
            "1 if z_l values are lower limits",
        )
        primary_header["GRIDPREC"] = meta["grid_precision"]

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
                hist = self.get_distribution(i, "redshift", bin_edges, normalize=True)
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
