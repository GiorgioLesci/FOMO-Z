"""
Calibrate color--color grid selections on a reference sample.

Compute weighted bin probabilities, apply threshold and significance filters,
and provide reference-sample metrics, bootstrap scatter, and file export.

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
    """
    Calibrate selections using color--color grids.

    Attributes
    ----------
    dict : dict
        Grid metadata, per-space object-bin mappings, and calibrated masks.
        ``dict[space]["mask"][i]`` is the grid mask for threshold index i.
    C, P, F : numpy.ndarray
        Global completeness, purity, and false positive rate after final
        filtering, in SelectionDefinition.z_lim order.
    C_std, P_std, F_std : numpy.ndarray
        Bootstrap standard deviations, or NaN when not estimated.
    target_mask : numpy.ndarray
        Prior-weighted target memberships, with shape
        (n_thresholds, n_reference). Boolean for unit-weight point targets.
    bin_probability_min : numpy.ndarray
        Last accepted bin-probability threshold for each calibration,
        also used by the significance filter. NaN before calibration.
    color_spaces : list
        Pairs of colors defining the available grids.
    dict_steps : dict
        Empty placeholder; iteration storage is not implemented.
    """

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
        """
        Initialize color grids and reference target memberships.

        A bin's target probability is the sum of target-selection weights
        within that bin divided by the sum of reference prior weights.
        Target-selection weights already include the prior weights and the
        redshift membership (from point estimates or PDZs). Empty bins and
        bins with zero total prior weight receive probability zero.

        Parameters
        ----------
        dataset : Dataset
            Reference magnitudes, target properties, prior weights, and
            optional cumulative PDZs. Clean required target properties before constructing
            this object and avoid modifying the sample afterward.
            Magnitude uncertainties are not propagated by this binary method.
        selection_definition : SelectionDefinition
            Criteria defining reference targets at each supplied threshold.
            Additional rules are shared across thresholds. Their required
            properties are validated during construction.
        purity_min : float or one-dimensional array-like
            Requested global purity in [0, 1). A Python float is broadcast
            to all redshift thresholds (when provided); otherwise supply one value
            per redshift threshold.
        grid_precision : float, default=0.15
            Positive nominal grid spacing in magnitude units, shared by all
            colors. The helper constructs ``int(width / grid_precision) + 1``
            evenly spaced edges over each color range. Actual spacing can
            differ from grid_precision.
        alpha : float, default=1
            Threshold in [0, 1] for the posterior lower-tail probability
            used to filter bins after the iterative search.

            For each bin, the helper computes an effective trial count
            ``n_eff = sum(w)**2 / sum(w**2)`` and effective successes
            ``k_eff = p_sel * n_eff``. The filter uses a Beta distribution
            with parameters ``k_eff + 1`` and
            ``max(n_eff - k_eff, 0) + 1``. A populated bin is retained only
            if its posterior CDF at the last accepted bin-probability threshold
            is strictly less than alpha. For unweighted binary targets,
            these effective counts reduce to ordinary counts. With weights
            or probabilistic targets, this is an effective-count approximation
            to the binomial/uniform-prior construction.

            Smaller alpha applies a stricter filter. Zero rejects all bins.
            One disables the significance filter entirely, without evaluating
            the Beta CDF. For alpha below one, filtering uses the last accepted
            bin-probability threshold.
        step : float, default=0.001
            Positive decrement of the bin-probability threshold, initially
            one. Smaller values sample the relaxation more finely.
            Nonfinite or non-positive values raise ValueError.
        do_bootstrap : bool, default=True
            Estimate metric scatter by resampling reference rows while
            keeping the final calibrated masks and grid boundaries fixed.
        n_bootstraps : int, default=1000
            Positive number of bootstrap realizations when enabled.
        verbose : bool, default=True
            Display progress messages and progress bars.

        Raises
        ------
        ValueError
            If purity thresholds have incompatible length or lie outside
            [0, 1), alpha is nonfinite or outside [0, 1], step or
            grid_precision is nonfinite or non-positive, or required target
            properties are invalid. Grid and target-evaluation errors propagate.

        Notes
        -----
        Color limits are the 0.0001 and 0.9999 quantiles of finite reference
        color values. Grids and object-bin indices are built at initialization.
        Objects outside a grid or with missing colors cannot be selected
        through that space, but may be selected through another space.
        All color--color selections are combined through logical OR.

        The search lowers a common bin-probability threshold, accepting bins
        with ``p_sel >= threshold``. Completeness cannot decrease as this
        unfiltered region expands.

        Once non-empty selections are reached, it stops at the first trial below
        the required global purity and retains the last accepted masks and
        threshold. If no trial was accepted, it raises ValueError. Reaching
        a non-positive threshold without crossing the purity requirement
        also raises ValueError, preserving the existing exhaustion behavior.
        These search errors expose ``calibration_failure``: either
        ``"no_accepted_selection"`` or ``"purity_not_crossed"``. Their
        ``calibration_purity`` attribute records the last measured purity,
        or NaN if no non-empty trial was evaluated. This diagnostic does not
        change the zero-denominator convention for stored metrics.

        Significance filtering is applied only after this search. It can
        change the global purity of the OR union in either direction.

        The global metrics use weighted counts: completeness is
        ``TP / (TP + FN)``, purity is ``TP / (TP + FP)``, and failure rate
        is ``FP / (FP + TN)``. Failure rate is the fraction of non-targets
        selected, not the contaminant fraction among selected objects.
        A metric with zero denominator is stored as zero.
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
        if not np.isfinite(self.grid_precision) or self.grid_precision <= 0:
            raise ValueError("grid_precision must be finite and strictly positive.")

        if not np.isfinite(alpha) or alpha < 0 or alpha > 1:
            raise ValueError(
                f"alpha must range between 0 and 1. The current value is {alpha}."
            )
        self.alpha = alpha

        if not np.isfinite(step) or step <= 0:
            raise ValueError("step must be finite and strictly positive.")
        self.step = float(step)
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
        self.bin_probability_min = np.full(len(self.td.z_lim), np.nan)

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

    def _process_for_one_z(self, args):
        """
        Calibrate and filter the grids for one target threshold.

        Parameters
        ----------
        args : tuple
            ``(i_z, store_iterations, cpu_number)``. Only i_z is used;
            iteration storage and multiprocessing remain unimplemented.

        Returns
        -------
        None
            Update per-space masks, bin_probability_min, and C, P, F.

        Raises
        ------
        ValueError
            If no non-empty trial is accepted or the search reaches zero
            without crossing the requested purity. These errors expose
            ``calibration_failure`` and ``calibration_purity`` attributes.
            The latter is NaN when every evaluated selection was empty.

        Notes
        -----
        Empty or zero-weight trials are skipped. The first non-empty trial
        below the required purity stops the search. The last accepted masks
        and threshold are retained together. With alpha below one, populated
        bins are filtered using the Beta CDF at that accepted threshold.
        With alpha equal to one, significance filtering is bypassed.

        Final metrics are recomputed. A UserWarning is emitted if significance
        filtering produces an empty selection or lowers purity below the
        requirement; no second calibration is performed.
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

        # Keep the last accepted masks and their bin threshold together.
        prob_min_bin = 1.0
        accepted_threshold = None
        last_purity = np.nan

        while prob_min_bin > 0.0:
            for cs in self.color_spaces:
                dict_stats[cs]["mask"][i_z] = (
                    dict_stats[cs]["prob"] >= prob_min_bin
                ) & (dict_stats[cs]["n"] > 0)

            mask = self.get_mask(i_z, dictionary=dict_stats)
            TP, FP, FN, TN = compute_metrics(mask, self.target_mask[i_z], self.weights)

            # Empty (or zero-weight) initial selections do not constrain purity.
            if TP + FP <= 0:
                prob_min_bin -= self.step
                continue

            P = TP / (TP + FP)
            C = TP / (TP + FN) if TP + FN > 0 else 0.0
            last_purity = P
            if P < self.thresholded_quantity_min[i_z]:
                break

            accepted_threshold = prob_min_bin
            self.P[i_z] = P
            self.C[i_z] = C
            self.F[i_z] = FP / (FP + TN) if FP + TN > 0 else 0.0
            for cs in self.color_spaces:
                self.dict[cs]["mask"][i_z] = dict_stats[cs]["mask"][i_z].copy()
            prob_min_bin -= self.step

        if accepted_threshold is None:
            error = ValueError(
                f"Calibration stopped without accepting a non-empty selection at purity_min="
                f"{self.thresholded_quantity_min[i_z]} at z_lim={self.td.z_lim[i_z]}."
            )
            error.calibration_failure = "no_accepted_selection"
            error.calibration_purity = float(last_purity)
            raise error

        # Preserve the existing exhaustion error used by calibration callers.
        if prob_min_bin <= 0:
            error = ValueError(
                "The bin-probability search reached zero without crossing "
                "the requested global purity. "
                f"Current P, C, F: {self.P[i_z]}, {self.C[i_z]}, {self.F[i_z]}."
            )
            error.calibration_failure = "purity_not_crossed"
            error.calibration_purity = float(self.P[i_z])
            raise error

        self.bin_probability_min[i_z] = accepted_threshold

        # alpha=1 explicitly disables significance filtering.
        if self.alpha < 1:
            for cs in self.color_spaces:
                k_eff = dict_stats[cs]["k"]
                n_eff = dict_stats[cs]["n"]
                significant = (n_eff > 0) & (
                    beta.cdf(
                        accepted_threshold,
                        k_eff + 1,
                        np.maximum(n_eff - k_eff, 0) + 1,
                    )
                    < self.alpha
                )
                self.dict[cs]["mask"][i_z] &= significant

        mask = self.get_mask(i_z)
        TP, FP, FN, TN = compute_metrics(mask, self.target_mask[i_z], self.weights)
        self.P[i_z] = TP / (TP + FP) if TP + FP > 0 else 0.0
        self.C[i_z] = TP / (TP + FN) if TP + FN > 0 else 0.0
        self.F[i_z] = FP / (FP + TN) if FP + TN > 0 else 0.0
        if self.alpha < 1 and (
            TP + FP <= 0 or self.P[i_z] < self.thresholded_quantity_min[i_z]
        ):
            warnings.warn(
                f"After significance filtering at z_lim={self.td.z_lim[i_z]}, "
                f"the selection is empty or its purity ({self.P[i_z]:.6g}) "
                f"is below purity_min ({self.thresholded_quantity_min[i_z]:.6g}). "
                "The filtered masks are retained and metrics have been recomputed.",
                UserWarning,
                stacklevel=2,
            )

    def _estimate_errors(self):
        """
        Estimate metric scatter with fixed-grid bootstrap resampling.

        Draw reference rows with replacement, preserving the sample size.
        Weights and cumulative PDZs follow the sampled rows, and target rules
        are reevaluated. Grid boundaries and calibrated masks remain fixed;
        bin probabilities and the selection calibration are not refitted.

        Returns
        -------
        None
            Set C_std, P_std, and F_std to the population standard deviations
            (ddof=0) over realizations. Original-sample C, P, and F remain
            unchanged; bootstrap means are not stored.

        Raises
        ------
        ValueError
            If the sum of calibrated completeness values is zero.

        Notes
        -----
        Uses NumPy's global random state. The resulting scatter is conditional
        on the calibrated masks. It does not include calibration uncertainty
        or sample variance between independent fields.
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
        """
        Calibrate each threshold, filter bins, and compute global metrics.

        Thresholds are processed sequentially. The objective is fixed to
        expanding the color selection subject to a requested global purity.
        No interpolation between threshold masks is performed.

        Returns
        -------
        None
            Update the per-space masks in dict and global metrics C, P,
            and F. If do_bootstrap is True, also update C_std, P_std, and
            F_std; otherwise their initial NaN values remain.

        Raises
        ------
        ValueError
            If no non-empty trial is accepted, the threshold search is
            exhausted, or bootstrap is requested without any nonzero
            calibrated completeness.

        Notes
        -----
        Multiprocessing and iteration-history storage are not implemented.
        The purity requirement applies before significance filtering. If
        filtering produces an empty selection or a lower purity, a warning
        is emitted and the final metrics describe the filtered masks.
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

    def get_distribution(
        self,
        zl_index=0,
        variable="redshift",
        bin_edges=np.linspace(0.0, 1.0, 10),
        normalize=True,
    ):
        """
        Compute a property distribution of color-selected reference objects.

        Combine reference prior weights with the color-selection mask.
        Redshift distributions always use reference PDZs when available.
        Otherwise, use point estimates. Other properties use point values.

        Parameters
        ----------
        zl_index : int
            Index of the threshold in SelectionDefinition.z_lim. If no
            redshift criteria were defined, set it to default (i.e. zero).
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

        The histogram includes all color-selected objects, including
        contaminants, weighted by their prior weights. It does not additionally
        apply the target-membership mask. Weight outside the requested bins
        is excluded and reported with a warning. Normalization requires
        positive weight within the range.
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
            classification_weights=self.get_mask(zl_index, color_space="all"),
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

        This FITS file is supposed to be read by the Classifier class.
        Other files can also be written by this method (see below).

        Parameters
        ----------
        output_dir : str
            Path to the directory where the file will be saved.
        file_name : str
            Base name of the output file (without extension).
            The output file is named ``FOMO-Z_{file_name}.fits``.
        reference_sample_name : str, default=""
            Name of the reference sample, stored in the FITS header.
        notes : str, default=""
            Additional notes stored in the FITS header.
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
        The output directory is created if needed. Existing files with these
        names are overwritten. Metrics columns are z_lim, C, P, F,
        C_std, P_std, and F_std; disabled bootstrap gives NaN uncertainties.
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
            for axis, color in enumerate(cs, start=1):
                low, high = meta["color_limits"][color]
                mask_hdu.header[f"C{axis}MIN"] = float(low)
                mask_hdu.header[f"C{axis}MAX"] = float(high)
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
        Return the available color--color space definitions.

        Returns
        -------
        list of tuple
            Keys of the form ``((band1, band2), (band3, band4))``, excluding
            metadata. Keys are available immediately after initialization;
            their presence does not imply calibration has been performed.
        """
        return [key for key in list(self.dict.keys()) if key != "metadata"]

    def get_mask(self, zl_index, color_space="all", dictionary=None):
        """
        Combine calibrated grid masks into an object-selection mask.

        Parameters
        ----------
        zl_index : int
            Index of the target threshold in SelectionDefinition.z_lim.
        color_space : str or tuple, default="all"
            ``"all"`` combines all spaces through logical OR. Alternatively,
            supply one space key returned by get_keys, such as
            ``(("g", "r"), ("r", "i"))``.
        dictionary : dict or None, default=None
            Grid masks and object-bin mappings. None uses self.dict.
            An alternative dictionary must contain the requested space keys
            and aligned valid flags, bin indices, and masks. Internally this
            permits evaluation on trial grids and bootstrap samples.

        Returns
        -------
        numpy.ndarray of bool
            One value per object represented by the dictionary's mappings.
            Objects pass if they occupy a selected bin in any requested
            space. Objects outside a grid or with unavailable colors are
            excluded from that space. No prior weights or target memberships
            are multiplied into this boolean photometric mask.

        Raises
        ------
        ValueError
            If color_space is not an available key or ``"all"``.
        KeyError
            If the requested threshold mask or a required dictionary entry
            is missing, for example before calibration.

        Notes
        -----
        This method uses existing bin indices; it does not bin new photometry.
        For self.dict, results are aligned with the reference sample.
        No interpolation between redshift thresholds is performed.
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
