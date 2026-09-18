"""
Derive probability point estimates.

This module derives point-estimate probabilities based on color grids,
given any condition on redshift or any other galaxy properties. The reference
dataset defines a grid of objects with known colors, and for each test object,
a probability is estimated from the fraction of reference objects
that are color-consistent with it within its photometric errors.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import warnings
import time
import multiprocessing as mp
import numpy as np
from numba import njit
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

from ..auxiliary.statistics import compute_metrics, get_color_color_mask
from ..data.dataset import Dataset
from ..data.selection_definition import SelectionDefinition
from ..color_selection.calibration_grid import CalibrationGrid


@njit(cache=True, fastmath=False, nogil=True)
def _matching_indices_numba(
    reference_colors,
    reference_errors,
    test_colors,
    test_errors,
    combine_errors,
):
    """
    Return matching reference indices using a compiled sequential scan.

    Parameters
    ----------
    reference_colors, reference_errors : numpy.ndarray
        Reference matrices with shape (n_colors, n_reference).
    test_colors, test_errors : numpy.ndarray
        Color values and errors of one test object, with shape (n_colors,).
    combine_errors : bool
        Whether to combine test and reference errors in quadrature.
        Otherwise, use only the test errors.

    Returns
    -------
    numpy.ndarray
        Matching reference indices in ascending order.

    Notes
    -----
    Test colors with a NaN value or error impose no constraint.
    A reference is rejected as soon as one usable color fails.
    No reference weights or target memberships are used here.
    """
    n_reference = reference_colors.shape[1]

    # Determine usable colors once for this test object
    usable = np.flatnonzero(
        ~np.isnan(test_colors) & ~np.isnan(test_errors)
    )

    # Allocate enough space for the maximum possible number of matches
    indices = np.empty(n_reference, dtype=np.int64)
    n_matches = 0

    for j in range(n_reference):
        accepted = True

        for position in range(usable.size):
            k = usable[position]

            difference = abs(
                reference_colors[k, j] - test_colors[k]
            )

            if combine_errors:
                tolerance = np.sqrt(
                    test_errors[k] ** 2
                    + reference_errors[k, j] ** 2
                )
            else:
                tolerance = test_errors[k]

            # Keep this form:
            # "difference > tolerance" would not reject NaNs
            if not (difference <= tolerance):
                accepted = False
                break

        if accepted:
            indices[n_matches] = j
            n_matches += 1

    # Return only the populated portion of the buffer.
    return indices[:n_matches]


class PointProbability:
    """Get point-estimate probabilities."""

    def __init__(
        self,
        selection_definition: SelectionDefinition,
        reference_dataset: Dataset,
        test_dataset: Dataset = None,
        verbose=True,
    ):
        r"""
        Initialize a color-based estimator of target-membership probabilities.

        For each test object, identify reference objects with compatible
        colors and estimate the probability of satisfying
        `selection_definition` from their weighted target memberships.
        Call `process` to compute the estimates; initialization only
        validates the input datasets and constructs the reference target mask.

        Parameters
        ----------
        selection_definition : SelectionDefinition
            Definition of the target population, such as objects above a
            redshift threshold, within a redshift interval, or satisfying
            additional property-based conditions. Exactly one `z_lim`
            value is supported.
        reference_dataset : Dataset
            Reference sample providing photometry, target information,
            and per-object weights. Its DataFrame must contain magnitude
            uncertainties in columns named ``<band>_error`` for every band,
            and finite, strictly positive weights in ``weights``.

            Target memberships are obtained from the reference properties
            or, when available, from the integrated redshift PDFs.
        test_dataset : Dataset, optional
            Sample for which probabilities are estimated. Its ordered
            color list must match that of `reference_dataset`, and its
            DataFrame must contain ``<band>_error`` columns.

            Test-sample weights and redshift PDFs are not used to estimate
            its probabilities. If None, the code uses `reference_dataset`
            as the test sample and applies the self-test matching convention
            described below.
        verbose : bool, default=True
            Whether to display progress bars and timing information.
            This option does not suppress warnings.

        Notes
        -----
        For reference object j, let q_j denote its target membership:
        either a binary indicator or a probability derived from its
        redshift PDF and any additional selection conditions. Let w_j
        denote its reference weight. The stored `target_mask` already
        includes that weight:

        .. math::
            t_j = w_j q_j.

        For a set of color-compatible references M_i, `process` estimates

        .. math::
            P_i = \frac{\sum_{j \in M_i} t_j}
                        {\sum_{j \in M_i} w_j}.

        Reference weights affect this estimate, but do not affect which
        objects satisfy the color-matching conditions.

        Color errors are computed by adding the two magnitude errors
        in quadrature, assuming uncorrelated errors between those bands.
        Matching applies separate constraints to each selected color;
        it does not model covariance between colors sharing a band.

        If `test_dataset` is explicitly supplied, matching tolerances
        combine test and reference color errors in quadrature. This
        assumes independent measurements. The same rule applies even
        if the explicitly supplied dataset is `reference_dataset` itself.

        If `test_dataset` is None, each matching tolerance is instead
        the test object's color error alone. Reference color errors
        are not added in this mode. This is a self-test convention,
        not the uncertainty on the difference between two independent
        reference objects.

        Self-test matching does not exclude the test object from its
        own reference matches. Consequently, this mode is not an
        independent validation or a leave-one-out evaluation.

        A test color is usable if neither its value nor its error is
        NaN. Missing test colors impose no constraint. `process` can
        require a minimum number of usable colors.

        The estimator provides probabilities of the events defined by
        `selection_definition`, not a probability density at an exact
        value of a continuous property.

        See Also
        --------
        select_features_grid : Select colors before probability estimation.
        process : Compute probabilities and their approximate standard errors.
        """
        if not isinstance(selection_definition, SelectionDefinition):
            raise TypeError(
                f"'selection_definition' must be a SelectionDefinition instance, got {type(selection_definition).__name__}."
            )
        if not isinstance(reference_dataset, Dataset):
            raise TypeError(
                f"'reference_dataset' must be a Dataset instance, got {type(reference_dataset).__name__}."
            )
        if test_dataset is not None and not isinstance(test_dataset, Dataset):
            raise TypeError(
                f"'test_dataset' must be a Dataset instance or None, got {type(test_dataset).__name__}."
            )

        # If no test dataset is supplied, reuse the reference sample
        # and apply the test-error-only matching convention.
        test_dataset_was_given = test_dataset is not None
        if not test_dataset_was_given:
            test_dataset = reference_dataset
        self._combine_reference_errors = test_dataset_was_given

        if test_dataset_was_given:
            if test_dataset.CDZs is not None:
                warnings.warn(
                    "PDZs were set in test_dataset, but they are not used by the PointProbability class."
                )
            if not np.allclose(np.asarray(test_dataset.dataset["weights"].values), 1.0):
                warnings.warn(
                    "Weights were set in test_dataset, but they are not used by the PointProbability class."
                )

        if len(reference_dataset.colors) != len(test_dataset.colors):
            raise ValueError(
                "The sets of colors in the reference and test datasets have different lengths."
            )
        for i in range(len(reference_dataset.colors)):
            if reference_dataset.colors[i] != test_dataset.colors[i]:
                raise ValueError(
                    "Different sets of colors in the reference and test datasets. The two lists of colors must match element by element. Provide the same band list to both Dataset objects."
                )
        for band in reference_dataset.bands:
            if band + "_error" not in reference_dataset.dataset.columns:
                raise ValueError(
                    f"Uncertainties on {band} are not defined. Include them in the '{band}_error' column within the DataFrame of reference_dataset."
                )
        for band in test_dataset.bands:
            if band + "_error" not in test_dataset.dataset.columns:
                raise ValueError(
                    f"Uncertainties on {band} are not defined. Include them in the '{band}_error' column within the DataFrame of test_dataset."
                )

        if len(selection_definition.z_lim) != 1:
            raise ValueError("Only one z_lim value is allowed in selection_definition.")

        # Create the target mask
        self.target_mask = selection_definition._get_target_mask(
            reference_dataset.dataset,
            selection_definition.z_lim,
            reference_dataset.CDZs,
            reference_dataset.PDZ_x,
        )[0]

        # Set other class variables
        self.reference_dataset = reference_dataset
        self.test_dataset = test_dataset
        self.selection_definition = selection_definition
        self.verbose = verbose
        self.colors = None
        self.probability_dictionary = {}

    def _set_dictionaries(self):
        r"""
        Build and store the color/error/weight matrices for reference and test data.

        Based on `self._build_color_dict`, this populates `self.reference_dict` and `self.test_dict`
        with the color matrices (and, respectively, weights and photometric color errors)
        computed from `self.reference_dataset` and `self.test_dataset`, using the colors
        currently stored in `self.colors` (either selected via feature selection, or
        set to all available colors as a fallback). Color errors are computed for both
        the reference and the test sample; whether they are actually combined when
        matching test and reference objects depends on `self._combine_reference_errors`
        (see `_matching_indices` and the class docstring).

        This method must be called after `self.colors` has been set, and before any
        method relying on `self.reference_dict` or `self.test_dict`.
        """
        self.N_reference_objects = len(self.reference_dataset.dataset)
        self.N_test_objects = len(self.test_dataset.dataset)
        self.reference_dict = self._build_color_dict(
            self.reference_dataset.dataset, True, True
        )
        self.test_dict = self._build_color_dict(self.test_dataset.dataset, True, False)

        # Store each reference object's colors contiguously in memory.
        # Shapes remain (n_colors, n_reference).
        self.reference_dict["colors"] = np.asfortranarray(
            self.reference_dict["colors"]
        )
        self.reference_dict["errors"] = np.asfortranarray(
            self.reference_dict["errors"]
        )

    def _build_color_dict(self, dataset, include_errors, include_weights):
        r"""
        Build a matrix of colors (and optionally their errors and weights) from a Dataset.

        For each color in `self.colors`, the corresponding color array is computed as the
        difference between the two magnitude columns it refers to, and stored as a row
        in a `(n_colors, n_objects)` matrix. If `include_errors` is True, a matrix of
        the same shape is also built, with the propagated color errors obtained by
        summing the two color errors in quadrature (missing magnitude errors
        propagate as NaN in the corresponding color error).

        Parameters
        ----------
        dataset : Dataset
            The `Dataset` object (reference or test) from which colors are computed.
        include_errors : bool
            If True, also compute and store the propagated color errors, using the
            magnitude error columns named `<band>_error`.
        include_weights : bool
            If True, also store the per-object weights from `dataset["weights"]`.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - `"colors"` : numpy.ndarray of shape `(n_colors, n_objects)`
              Matrix of colors, one row per color in `self.colors`, one column per
              object in `dataset`.
            - `"errors"` : numpy.ndarray of shape `(n_colors, n_objects)`, optional
              Matrix of propagated color errors, same layout as `"colors"`.
              Only present if `include_errors` is True.
            - `"weights"` : numpy.ndarray of shape `(n_objects,)`, optional
              Per-object weights. Only present if `include_weights` is True.
        """
        n_colors = len(self.colors)
        n_obj = len(dataset)
        color_matrix = np.empty((n_colors, n_obj))

        if include_errors:
            error_matrix = np.empty((n_colors, n_obj))

        for k, color in enumerate(self.colors):
            band1, band2 = color
            color_matrix[k] = dataset[band1].values - dataset[band2].values
            if include_errors:
                err1 = dataset[band1 + "_error"].values
                err2 = dataset[band2 + "_error"].values
                error_matrix[k] = np.sqrt(err1**2 + err2**2)

        weights = np.asarray(dataset["weights"].values) if include_weights else None

        result = {"colors": color_matrix}
        if include_errors:
            result["errors"] = error_matrix
        if weights is not None:
            result["weights"] = weights
        return result

    def _matching_indices(self, i):
        """
        Return indices of references compatible with test object i.

        Parameters
        ----------
        i : int
            Positional index of the test object.

        Returns
        -------
        numpy.ndarray
            Matching reference indices in ascending order. Returns an
            empty array if no reference matches, or all reference indices
            if the test object has no usable colors.

        Notes
        -----
        Delegates matching to the compiled `_matching_indices_numba`
        function. Requires `_set_dictionaries` to have been called.

        A test color is usable when its value and error are non-NaN.
        Every usable color must satisfy the matching tolerance.

        If `_combine_reference_errors` is True, test and reference
        errors are combined in quadrature. Otherwise, only test errors
        are used. Reference weights do not affect matching.

        `process` enforces the minimum number of usable colors before
        calling this method.
        """
        return _matching_indices_numba(
            self.reference_dict["colors"],
            self.reference_dict["errors"],
            self.test_dict["colors"][:, i],
            self.test_dict["errors"][:, i],
            self._combine_reference_errors,
        )

    def select_features_grid(
        self,
        tolerance=0.1,
        purity_min=0.9,
        grid_precision=0.15,
        step=0.001,
    ):
        r"""
        Select the most informative colors using a grid-based color selection.

        This performs feature selection by calibrating a color selection with the
        `CalibrationGrid` class (from the `color_selection` subpackage), requiring
        a minimum global purity of `purity_min`.
        The `reference_dataset` and `selection_definition` parameters given as input
        to the constructor of this class are also given in input to `CalibrationGrid`.
        The `test_dataset` parameter is not used here.

        After this initial calibration, colors are removed one at a time in an
        iterative backward elimination process. At each step, among the colors still
        in the selection, the least informative one is identified as the color whose
        removal causes the smallest drop in global completeness (the quantity this
        procedure aims to preserve). Purity, instead, is the constrained quantity.
        Removal of a color is accepted, and the color permanently excluded from the
        selection, only if the resulting global purity and completeness both differ
        by no more than `tolerance` from the values obtained with the full set of
        colors (i.e. before any color was removed).
        The procedure stops as soon as no remaining color can be removed while keeping
        both variations within `tolerance`.

        The selected colors are stored in `self.colors`, to be used by subsequent
        calls to `process`.

        Since evaluating all possible color subsets would be combinatorially
        expensive, this greedy backward elimination is used instead; it is not
        guaranteed to find the globally optimal subset of colors, but it is a
        standard, tractable compromise. This is one of possibly several feature
        selection strategies; other approaches may be added as separate methods in
        the future.

        Parameters
        ----------
        tolerance : float, default=0.1
            Maximum allowed relative variation in global purity and completeness, at
            each elimination step, for a color to be excluded from the selection.
            This is a relative tolerance, not a variation in percentage points: e.g.
            a value of 0.1 means the purity/completeness can change by at most 10%
            of its current value, not by 10 percentage points. Note that, since this
            is a relative tolerance, its effective strictness depends on how close
            purity and completeness are to zero: near-zero values make the tolerance
            very permissive in absolute terms.
        purity_min : float, default=0.9
            Required minimum value for the global purity of the initial grid-based
            selection. Default (recommended) is 90%.
        grid_precision : float, default=0.15
            Precision of the color-color grids. A smaller value yields a finer,
            more accurate grid at the cost of increased computation, RAM usage, and
            storage requirements.
        step : float, default=0.001
            Threshold reduction step applied when refining the grid selection.
        """
        if not isinstance(purity_min, float):
            raise ValueError("purity_min must be a float.")

        # Calibrate the color selection
        start = time.perf_counter()
        calibration = CalibrationGrid(
            self.reference_dataset,
            self.selection_definition,
            purity_min,
            grid_precision=grid_precision,
            step=step,
            do_bootstrap=False,
            verbose=False,
        )
        calibration.process()

        purity_baseline, completeness_baseline = calibration.P[0], calibration.C[0]

        if self.verbose:
            elapsed = time.perf_counter() - start
            print(
                f"CalibrationGrid processing complete. Employed time: {elapsed:.2f} s."
            )
            print(
                f"\nCalibrationGrid's purity and completeness (respectively): {purity_baseline:.2f}, {completeness_baseline:.2f}"
            )

        # Get all color spaces and colors used in CalibrationGrid
        color_spaces = (
            calibration.get_keys()
        )  # if u, g, r bands, then [(('u', 'g'), ('u', 'r')), (('u', 'g'), ('g', 'r')), (('u', 'r'), ('g', 'r'))]

        all_colors = list(
            dict.fromkeys(color for pair in color_spaces for color in pair)
        )

        # Initialize best_colors and discarded_colors
        best_colors = list(all_colors)
        discarded_colors = []

        # Precompute which grids (color-color pairs) involve each color
        grids_by_color = {
            color: [pair for pair in color_spaces if color in pair]
            for color in all_colors
        }

        # Grids currently valid, i.e. both their colors are still in best_colors.
        # Starts as the full set of grids, since best_colors starts as all_colors.
        active_grids = list(color_spaces)

        # Iteratively remove, one at a time, the color whose removal causes the
        # smallest drop in completeness, as long as removing it keeps both purity
        # and completeness within `tolerance` (relative) of the fixed baseline.
        start = time.perf_counter()
        iteration_number = 0
        last_accepted_purity, last_accepted_completeness = (
            purity_baseline,
            completeness_baseline,
        )

        while len(best_colors) > 2:

            candidate_removed = None
            candidate_trial_grids = None
            candidate_purity = None
            candidate_completeness = None
            candidate_completeness_drop = None

            for color in best_colors:
                # Grids that survive if color is removed: active grids minus those
                # that involve color (a grid needs both its colors present)
                trial_grids = [
                    g for g in active_grids if g not in grids_by_color[color]
                ]

                mask_trial = get_color_color_mask(0, trial_grids, calibration.dict)
                TP, FP, FN, TN = compute_metrics(
                    mask_trial, calibration.target_mask[0], calibration.weights
                )
                purity_trial = TP / (TP + FP) if TP + FP > 0 else 0.0
                completeness_trial = TP / (TP + FN) if TP + FN > 0 else 0.0

                completeness_drop = completeness_baseline - completeness_trial

                if (
                    candidate_completeness_drop is None
                    or completeness_drop < candidate_completeness_drop
                ):
                    candidate_removed = color
                    candidate_trial_grids = trial_grids
                    candidate_purity = purity_trial
                    candidate_completeness = completeness_trial
                    candidate_completeness_drop = completeness_drop

            purity_change = abs(candidate_purity - purity_baseline) / purity_baseline
            completeness_change = (
                abs(candidate_completeness - completeness_baseline)
                / completeness_baseline
            )

            if purity_change <= tolerance and completeness_change <= tolerance:
                best_colors = [c for c in best_colors if c != candidate_removed]
                discarded_colors.append(candidate_removed)
                active_grids = candidate_trial_grids
                iteration_number += 1
                last_accepted_purity = candidate_purity
                last_accepted_completeness = candidate_completeness
            else:
                break

        if self.verbose:
            print(
                f"Purity and completeness after the selection of features : {last_accepted_purity:.2f}, {last_accepted_completeness:.2f}"
            )

            elapsed = time.perf_counter() - start
            print(
                f"Feature selection complete after {iteration_number} iterations. Employed time: {elapsed:.2f} s."
            )

        # Store the most informative colors
        if iteration_number == 0:
            warnings.warn(
                "No feature could be removed within tolerance. Using all available features."
            )
            self.colors = self.reference_dataset.colors
            self.discarded_colors = []
        else:
            self.colors = best_colors
            self.discarded_colors = discarded_colors
            if self.verbose:
                print(f"\nSelected features (self.colors):\n{best_colors}")
                print(
                    f"\nDiscarded features (self.discarded_colors):\n{discarded_colors}"
                )

    def process(
        self,
        min_colors_used=2,
        silence_warnings=False,
        cpu_number=None,
        block_size=256,
    ):
        r"""
        Compute weighted target-membership probabilities for all test objects.

        Use the colors stored in `self.colors`, or all reference colors
        if feature selection has not been performed. For each test
        object, find the reference objects satisfying every usable
        color constraint and combine their weighted target memberships.

        Parameters
        ----------
        min_colors_used : int, default=2
            Minimum number of usable test colors required to estimate
            a probability. A color is usable when neither its value
            nor its error is NaN.

            Objects below this threshold are skipped: their probability
            and error remain NaN, and their match count and weighted
            target sum remain zero.
        silence_warnings : bool, default=False
            Suppress explicit warnings emitted by this method about
            missing feature selection, insufficient usable colors,
            unmatched objects, and single positive-weight matches.
            Progress-bar visibility is controlled by `self.verbose`.
        cpu_number : int or None, optional
            Number of worker threads used for parallel processing.
            If None, use the number of logical CPUs reported by
            multiprocessing.cpu_count() minus two, with a minimum of one.
            Explicit values are capped at the same limit.
            If 1, run sequentially without creating worker threads.
        block_size : int, default=256
            Number of test objects per task. Progress is updated as
            blocks are collected. This does not allocate a matrix of
            pairwise comparisons for the entire block.

        Returns
        -------
        None
            Results are stored in `self.probability_dictionary`.
            Each entry is an array of shape ``(n_test,)``:

            ``prob``
                Weighted target-membership probability. NaN for skipped
                objects, objects without matches.
            ``prob_error``
                Approximate sampling standard error of ``prob``, as
                defined below. NaN if the probability is undefined
                or fewer than two matched references have positive weight.
            ``n_good``
                Sum of `self.target_mask` over matching references.
                This is already a weighted target-membership sum:
                reference weights are not applied a second time.
                Zero for skipped objects or objects without matches.
            ``n_all``
                Unweighted number of matching reference objects. Zero for
                skipped objects or objects without matches.
            ``n_colors_used``
                Number of non-NaN test colors with non-NaN errors,
                including for objects skipped by `min_colors_used`.

        Raises
        ------
        ValueError
            If `cpu_number` is neither None nor a positive integer,
            or if `block_size` is not a positive integer.

        Notes
        -----
        Let M_i be the set of references compatible with test object i.
        For each reference j, let w_j be its weight and q_j its target
        membership, which may be binary or derived from a redshift PDF.
        The stored target-mask value is t_j = w_j q_j.

        The probability estimate is

        .. math::
            W_i = \sum_{j \in M_i} w_j,
            \qquad
            P_i = \frac{\sum_{j \in M_i} t_j}{W_i}.

        Thus, ``n_good / n_all`` is generally not the probability:
        the denominator must be the sum of reference weights.

        For m_i > 1 matched references with positive weight, estimate
        the sampling variance of this ratio as

        .. math::
            \widehat{\sigma}_{P_i}^{\,2}
            = \frac{m_i}{m_i - 1}
              \frac{\sum_{j \in M_i}(t_j - P_i w_j)^2}{W_i^2}.

        This is an approximate variance estimate for independent
        reference objects with relative sampling or importance weights.
        The finite-sample correction recovers the usual squared
        standard error, s_i^2 / m_i, when all positive weights are equal,
        where s_i is the sample standard deviation of q_j with ddof=1.

        Both the probability and its estimated error are invariant
        under a common positive rescaling of the reference weights.

        The error describes sampling variation of the weighted mean
        of q_j. It does not include uncertainty in the weight calibration,
        correlations between references, or systematic errors in the
        input photometry or redshift PDFs. It is not a predictive
        uncertainty for an individual object's binary membership.

        Identical q_j values among all positive-weight matches produce
        a zero estimated error; this does not establish that the true
        probability is known exactly.

        When the reference sample is also the test sample, self-matches
        are retained. The resulting probabilities are not independent
        validation predictions.

        See Also
        --------
        _matching_indices : Identify color-compatible reference objects.
        select_features_grid : Select the colors used for matching.
        """
        if cpu_number is not None:
            if (
                isinstance(cpu_number, (bool, np.bool_))
                or not isinstance(cpu_number, (int, np.integer))
                or cpu_number < 1
            ):
                raise ValueError(
                    "cpu_number must be a positive integer or None."
                )

        if (
            isinstance(block_size, (bool, np.bool_))
            or not isinstance(block_size, (int, np.integer))
            or block_size < 1
        ):
            raise ValueError("block_size must be a positive integer.")

        max_workers = max(mp.cpu_count() - 2, 1)

        n_jobs = (
            max_workers
            if cpu_number is None
            else min(int(cpu_number), max_workers)
        )

        if self.colors is None:
            if not silence_warnings:
                warnings.warn(
                    "No feature selection was performed. Using all available features."
                )
            self.colors = self.reference_dataset.colors

        # Set the feature dictionaries
        self._set_dictionaries()

        # Select reference objects that are color-consistent with the i-th test object.
        # Then, assign a probability to each test object.
        prob = np.full(self.N_test_objects, np.nan)
        n_good = np.zeros(self.N_test_objects)
        n_all = np.zeros(self.N_test_objects)
        prob_error = np.full(self.N_test_objects, np.nan)
        n_colors_used = np.zeros(self.N_test_objects, dtype=int)

        # Compile/load the Numba specialization before starting workers.
        if self.N_test_objects > 0:
            self._matching_indices(0)

        def process_block(start):
            """Process a disjoint block and return its warning counts."""
            stop = min(start + block_size, self.N_test_objects)

            block_unmatched = 0
            block_single_match = 0
            block_below_min_colors = 0

            for i in range(start, stop):
                usable_i = (
                    ~np.isnan(self.test_dict["colors"][:, i])
                    & ~np.isnan(self.test_dict["errors"][:, i])
                )
                n_colors_used[i] = int(usable_i.sum())

                if n_colors_used[i] < min_colors_used:
                    block_below_min_colors += 1
                    continue

                # The compiled kernel releases the GIL during matching.
                indices = self._matching_indices(i)

                values = self.target_mask[indices]
                weights = self.reference_dict["weights"][indices]

                n = indices.size
                n_all[i] = n
                n_good[i] = values.sum()

                if n == 0:
                    block_unmatched += 1
                    continue

                # Dataset guarantees finite, strictly positive weights.
                weight_sum = weights.sum()
                prob[i] = n_good[i] / weight_sum

                if n > 1:
                    residuals = (
                        values - prob[i] * weights
                    ) / weight_sum

                    variance = (
                        n / (n - 1)
                        * np.sum(residuals * residuals)
                    )
                    prob_error[i] = np.sqrt(variance)
                else:
                    block_single_match += 1

            return (
                stop - start,
                block_unmatched,
                block_single_match,
                block_below_min_colors,
            )

        n_unmatched = 0
        n_single_match = 0
        n_below_min_colors = 0

        block_starts = range(
            0, self.N_test_objects, block_size
        )

        # With n_jobs=1, use ordinary map without creating worker threads
        pool_context = (
            ThreadPoolExecutor(max_workers=int(n_jobs))
            if n_jobs > 1
            else nullcontext()
        )

        with pool_context as executor:
            if executor is None:
                results = map(process_block, block_starts)
            else:
                results = executor.map(process_block, block_starts)

            # Only the calling thread updates progress and warning counts
            with tqdm(
                total=self.N_test_objects,
                desc="Estimating point-estimate probabilities",
                disable=not self.verbose,
            ) as progress:
                for (
                    n_processed,
                    block_unmatched,
                    block_single_match,
                    block_below_min_colors,
                ) in results:
                    n_unmatched += block_unmatched
                    n_single_match += block_single_match
                    n_below_min_colors += block_below_min_colors

                    progress.update(n_processed)

        if n_below_min_colors > 0 and not silence_warnings:
            warnings.warn(
                f"{n_below_min_colors} test object(s) had fewer than "
                f"min_colors_used={min_colors_used} usable colors; their 'prob' "
                "was set to NaN."
            )
        if n_unmatched > 0 and not silence_warnings:
            warnings.warn(
                f"{n_unmatched} test object(s) had no matching reference objects; "
                "their 'prob' and 'prob_error' were set to NaN."
            )
        if n_single_match > 0 and not silence_warnings:
            warnings.warn(
                f"{n_single_match} test object(s) had only one matching reference "
                "object with positive weight; their 'prob_error' was set to NaN."
            )

        self.probability_dictionary = {
            "prob": prob,
            "prob_error": prob_error,
            "n_good": n_good,
            "n_all": n_all,
            "n_colors_used": n_colors_used,
        }
