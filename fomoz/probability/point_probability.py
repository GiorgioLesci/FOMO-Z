"""
Derive probability point estimates.

This module derives point-estimate probabilities based on colors,
given any condition on any galaxy property. 
For each test object, a probability is estimated from the fraction
of reference objects that are color-consistent with it within the
photometric errors.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import warnings
import time
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

from ..auxiliary.statistics import compute_metrics, get_color_color_mask
from ..auxiliary.probability_helper import _matching_indices_numba
from ..data.dataset import Dataset
from ..data.selection_definition import SelectionDefinition
from ..binary_classification.grid_classification import GridClassification


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
        Initialize a color-based estimator of point-estimate probabilities.

        For each test object, identify reference objects with compatible
        colors and estimate the probability of satisfying
        `selection_definition` from their weighted target memberships.
        Call `process` to compute the probability estimates.

        Parameters
        ----------
        selection_definition : SelectionDefinition
            Definition of the target population, such as objects above a
            redshift threshold, within a redshift interval, or satisfying
            additional conditions on other physical properties.
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
            probabilities. If `None`, the code uses `reference_dataset`
            as the test sample and applies the self-test matching convention
            described below.
        verbose : bool, default=True
            Whether to display progress bars and timing information.
            This option does not suppress warnings.

        Notes
        -----
        Let Phi denote the reference sample and X the full set of colors
        (i.e. features). Feature selection retains a subset X_sel; if it is
        not performed, all available features are used. Currently available
        feature selection methods are: `select_features_grid`.

        For test object i, chi_i is the subset of X_sel containing the
        selected colors whose measurements and uncertainties are both non-NaN.
        The set of reference objects that are color-compatible with the test
        object i is defined as

        .. math::

            \Sigma_i =
            \left\{
                j \in \Phi :
                |c_{i,k} - c_{j,k}| \leq \epsilon_{ij,k}
                \quad \forall k \in \chi_i
            \right\},

        where c_{i,k} and c_{j,k} are the values of color k for the test
        and reference objects, respectively. Missing test colors impose
        no matching constraint.

        When `test_dataset` is explicitly supplied, the matching
        tolerance combines test and reference color uncertainties:

        .. math::

            \epsilon_{ij,k} =
            \sqrt{\sigma_{i,k}^{2} + \sigma_{j,k}^{2}}.

        For a color c_k = m_a - m_b, its uncertainty is computed as

        .. math::

            \sigma_k =
            \sqrt{\sigma_{m_a}^{2} + \sigma_{m_b}^{2}}.

        These expressions assume uncorrelated photometric errors
        between the two bands and between the test and reference
        measurements. Matching constraints are applied separately
        to each selected color; covariance between colors sharing
        a band is not modeled.

        If `test_dataset` is None or if it matches exactly `reference_dataset`,
        the reference sample is also used as the test sample, and
        matching tolerances use only the test object's color uncertainties.
        Thus, the tolerance differs from the expression above and reads as

        .. math::

            \epsilon_{ij,k} = \sigma_{i,k}.

        Reference uncertainties are not added in quadrature in this
        mode, and self-matches are retained. This is a self-test
        convention, not an independent validation or a leave-one-out
        evaluation. We stress that this behavior is obtained both
        when `test_dataset` is None and when the reference
        `Dataset` is explicitly supplied as `test_dataset`.

        Once Sigma_i has been identified, `process` estimates the
        probability that test object i satisfies the selection rules R,
        defined by `selection_definition`:

        .. math::

            p_i(\mathcal{R}) =
            \frac{
                \sum_{j \in \Sigma_i}
                w_{\mathrm{sel}}^{j}(\mathcal{R})
            }{
                \sum_{j \in \Sigma_i}
                w_{\mathrm{prior}}^{j}
            }.

        Here, w_prior is the reference object's prior weight, stored
        in the reference DataFrame's ``weights`` column. The selection
        weight w_sel is constructed by `SelectionDefinition` and stored
        in `target_mask`. It already combines the prior weight with
        target membership and must not be multiplied by the prior
        weight again.

        For a redshift selection, target membership is binary when
        evaluated using a point estimate. When reference redshift PDFs
        are supplied, membership is instead obtained from their
        cumulative probabilities over the requested redshift range.
        See `SelectionDefinition`.

        Thus, p_i is the prior-weighted mean target membership among
        the color-compatible reference objects. Prior weights and
        redshift PDFs supplied for test objects are not used in this
        estimate; only their reference-sample counterparts contribute.
        For a fixed selected feature set, reference weights do not
        enter the color-matching criterion.

        Objects with fewer than `min_colors_used` usable selected colors
        (see `process`), or without matching references, receive NaN
        probability estimates.

        Photometric uncertainties explicitly set the matching tolerance,
        whereas intrinsic scatter in the color-property relation enters
        through the reference sample itself. Objects with closely matching
        colors may therefore differ in target membership. The estimate
        reflects both measurement uncertainties and intrinsic diversity,
        without separating their contributions.

        The method therefore assigns each test object a probability
        of belonging to the target population defined by
        `selection_definition`, such as its probability of being
        star-forming or lying within a specified redshift interval.

        For a continuous property, avoid defining the target through
        an exact equality, such as z = 0.5. This event has zero probability
        under a continuous distribution, and exact matches in a reference
        catalog depend on numerical precision or rounding. Instead, define
        a finite interval, such as 0.45 < z < 0.55, or a threshold, such as
        z > 0.5. To estimate the redshift probability density, use
        `ProbabilityFunction`.

        The statistical uncertainty on the estimated probability is
        returned as ``prob_error``. For n_i = |Sigma_i| > 1 matched
        reference objects, it is computed as

        .. math::

            \widehat{\sigma}_{p_i}^{\,2}
            =
            \frac{n_i}{n_i - 1}
            \frac{
                \sum_{j \in \Sigma_i}
                \left[
                    w_{\mathrm{sel}}^{j}(\mathcal{R})
                    - p_i(\mathcal{R}) w_{\mathrm{prior}}^{j}
                \right]^2
            }{
                \left(
                    \sum_{j \in \Sigma_i} w_{\mathrm{prior}}^{j}
                \right)^2
            }.

        This is an approximate sampling variance of the weighted mean
        target membership, treating the matched reference objects as
        independent and their prior weights as fixed relative weights.
        ``prob_error`` is the square root of this variance.

        For equal positive prior weights, the expression reduces to
        the sample variance of target memberships divided by n_i.

        This uncertainty describes finite-reference-sample variation
        for the adopted matching set and selection rules. It does not
        propagate uncertainty in the matching set, feature selection,
        prior weights, or reference target memberships. In particular,
        using photometric errors for matching and reference PDZs for
        target membership does not constitute a full propagation of
        their uncertainties into ``prob_error``.

        The returned error is not the predictive dispersion of an
        individual object's binary membership. Identical target
        memberships among the matched references produce a zero
        estimated error, which does not imply that the underlying
        membership probability is known exactly.

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

        # Reuse the reference sample when no test dataset is supplied
        test_dataset_was_given = test_dataset is not None
        if not test_dataset_was_given:
            test_dataset = reference_dataset

        # Combine uncertainties only for distinct Dataset instances
        self.test_is_reference = test_dataset is reference_dataset
        self._combine_reference_errors = not self.test_is_reference

        if not self.test_is_reference:
            if test_dataset.CDZs is not None:
                warnings.warn(
                    "PDZs were set in test_dataset, but they are not used by the PointProbability class."
                )
            if not np.allclose(np.asarray(test_dataset.dataset["weights"].values), 1.0):
                warnings.warn(
                    "Prior weights were set in test_dataset, but they are not used by the PointProbability class."
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
            reference_dataset.z_gal_string,
            reference_dataset.missing_data,
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
        self.reference_dict["colors"] = np.asfortranarray(self.reference_dict["colors"])
        self.reference_dict["errors"] = np.asfortranarray(self.reference_dict["errors"])

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
        step=0.01,
        retry_calibration=True,
    ):
        r"""
        Select informative colors through `GridClassification` methods and backward elimination.

        Calibrate a selection in all color-color grids with `GridClassification`, then
        iteratively remove the colorproducing the smallest decrease in global completeness.
        Accept a removal only if both purity and completeness remain within
        `tolerance` of their fixed values from the initial calibration.

        Each grid's galaxy-selection mask is computed once. Integer coverage
        counts track how many active grids select each galaxy. These counts
        are used only to construct Boolean selection masks; all metrics
        retain the original reference weights and weighted target memberships.

        Parameters
        ----------
        tolerance : float, default=0.1
            Maximum relative change in purity and completeness with respect
            to the fixed initial calibration baseline.
        purity_min : float, default=0.9
            Minimum global purity requested for the initial calibration.
            May be increased automatically if `retry_calibration=True`.
        grid_precision : float, default=0.15
            Resolution of the color-color grids.
        step : float, default=0.01
            Threshold reduction step used during calibration.
        retry_calibration : bool, default=True
            If True, retry with a larger purity threshold if `purity_min`
            is too low (i.e., if the `GridClassification` calibration
            converges to a larger purity value). If False, the code raises
            an error if `purity_min` is too low.

        Returns
        -------
        None
            Store selected colors in `self.colors` and removed colors in
            `self.discarded_colors`.
        """
        if not isinstance(purity_min, float):
            raise ValueError("purity_min must be a float.")
        if not np.isfinite(purity_min) or not 0.0 < purity_min < 1.0:
            raise ValueError("purity_min must be finite and between 0 and 1.")

        if not isinstance(retry_calibration, (bool, np.bool_)):
            raise TypeError("retry_calibration must be a boolean.")

        start = time.perf_counter()

        initial_purity_min = purity_min
        current_purity_min = purity_min
        n_retries = 0

        while True:
            try:
                calibration = GridClassification(
                    self.reference_dataset,
                    self.selection_definition,
                    current_purity_min,
                    grid_precision=grid_precision,
                    step=step,
                    do_bootstrap=False,
                    verbose=False,
                )
                calibration.process()
                break

            except ValueError as error:
                # Retry only the recognized calibration failure.
                if not retry_calibration or not hasattr(error, "calibration_purity"):
                    raise

                achieved_purity = float(error.calibration_purity)
                new_purity_min = achieved_purity + 0.001

                if (
                    not np.isfinite(new_purity_min)
                    or new_purity_min >= 1.0
                    or new_purity_min <= current_purity_min
                ):
                    raise ValueError(
                        "Cannot retry feature selection: "
                        f"achieved purity={achieved_purity:.6f}, "
                        f"current purity_min={current_purity_min:.6f}, "
                        f"proposed purity_min={new_purity_min:.6f}. "
                        "The new threshold must increase and remain below 1."
                    ) from error

                current_purity_min = new_purity_min
                n_retries += 1

        purity_baseline = calibration.P[0]
        completeness_baseline = calibration.C[0]

        if self.verbose:
            elapsed = time.perf_counter() - start
            print(
                "GridClassification processing complete. "
                f"Employed time: {elapsed:.2f} s."
            )
            print(
                "\nGridClassification's purity and completeness "
                f"(respectively): {purity_baseline:.2f}, "
                f"{completeness_baseline:.2f}"
            )

        start = time.perf_counter()

        color_spaces = list(calibration.get_keys())

        # Preserve the original order, including candidate tie-breaking.
        all_colors = list(
            dict.fromkeys(color for pair in color_spaces for color in pair)
        )

        best_colors = list(all_colors)
        discarded_colors = []

        n_grids = len(color_spaces)
        n_objects = len(calibration.weights)

        # Cache each grid's Boolean selection mask once.
        # Use the existing function to preserve its exact bin semantics.
        grid_masks = np.empty(
            (n_grids, n_objects),
            dtype=bool,
        )

        for g, pair in enumerate(color_spaces):
            grid_masks[g] = get_color_color_mask(0, [pair], calibration.dict)

        # Indices of grids involving each color.
        grids_by_color = {
            color: [g for g, pair in enumerate(color_spaces) if color in pair]
            for color in all_colors
        }

        active_grids = np.ones(n_grids, dtype=bool)

        # Number of active grids selecting each galaxy.
        # These are NOT galaxy weights or target memberships.
        coverage = grid_masks.sum(axis=0, dtype=np.int64)

        # Keep the weighted statistical inputs unchanged.
        target_mask = calibration.target_mask[0]
        weights = calibration.weights

        iteration_number = 0
        last_accepted_purity = purity_baseline
        last_accepted_completeness = completeness_baseline

        while len(best_colors) > 2:
            candidate_removed = None
            candidate_removed_grids = None
            candidate_removed_coverage = None
            candidate_purity = None
            candidate_completeness = None
            candidate_completeness_drop = None

            for color in best_colors:
                # Removing this color invalidates only its active grids.
                removed_grids = [g for g in grids_by_color[color] if active_grids[g]]

                # Count how many selecting grids each galaxy would lose.
                # Sum one row at a time to avoid a large temporary slice.
                removed_coverage = np.zeros(
                    n_objects,
                    dtype=coverage.dtype,
                )

                for g in removed_grids:
                    removed_coverage += grid_masks[g]

                # Selected iff at least one selecting grid survives.
                mask_trial = coverage > removed_coverage

                # Preserve weighted metrics exactly as in the original code.
                TP, FP, FN, TN = compute_metrics(
                    mask_trial,
                    target_mask,
                    weights,
                )

                purity_trial = TP / (TP + FP) if TP + FP > 0 else 0.0
                completeness_trial = TP / (TP + FN) if TP + FN > 0 else 0.0

                completeness_drop = completeness_baseline - completeness_trial

                # Strict comparison preserves the original tie-breaking.
                if (
                    candidate_completeness_drop is None
                    or completeness_drop < candidate_completeness_drop
                ):
                    candidate_removed = color
                    candidate_removed_grids = removed_grids
                    candidate_removed_coverage = removed_coverage
                    candidate_purity = purity_trial
                    candidate_completeness = completeness_trial
                    candidate_completeness_drop = completeness_drop

            purity_change = abs(candidate_purity - purity_baseline) / purity_baseline
            completeness_change = (
                abs(candidate_completeness - completeness_baseline)
                / completeness_baseline
            )

            if purity_change <= tolerance and completeness_change <= tolerance:
                best_colors = [
                    color for color in best_colors if color != candidate_removed
                ]
                discarded_colors.append(candidate_removed)

                # Update coverage only after accepting the removal.
                coverage -= candidate_removed_coverage
                active_grids[candidate_removed_grids] = False

                iteration_number += 1
                last_accepted_purity = candidate_purity
                last_accepted_completeness = candidate_completeness
            else:
                break

        if self.verbose:
            print(
                "Purity and completeness after the selection of features : "
                f"{last_accepted_purity:.2f}, "
                f"{last_accepted_completeness:.2f}"
            )

            elapsed = time.perf_counter() - start
            print(
                "Feature selection complete after "
                f"{iteration_number} iterations. "
                f"Employed time: {elapsed:.2f} s."
            )

        if iteration_number == 0:
            warnings.warn(
                "No feature could be removed within tolerance. "
                "Using all available features."
            )
            self.colors = self.reference_dataset.colors
            self.discarded_colors = []
        else:
            self.colors = best_colors
            self.discarded_colors = discarded_colors

            if self.verbose:
                print(f"\nSelected features (self.colors):\n{best_colors}")
                print(
                    "\nDiscarded features (self.discarded_colors):\n"
                    f"{discarded_colors}"
                )

        if n_retries:
            warnings.warn(
                f"Calibration required {n_retries} retries. "
                f"The requested purity_min increased from "
                f"{initial_purity_min:.6f} to "
                f"{current_purity_min:.6f}. "
                "Feature selection completed successfully.",
                stacklevel=2,
            )

    def process(
        self,
        min_colors_used=2,
        silence_warnings=False,
        cpu_number=None,
        block_size=256,
    ):
        r"""
        Compute weighted probabilities for all test objects.

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
                Weighted membership probability. NaN for skipped
                objects, objects without matches.
            ``prob_error``
                Approximate sampling standard error of ``prob``, as
                defined in the constructor's documentation. NaN if the
                probability is undefined or fewer than two matched
                references have positive weight.
            ``n_good``
                Sum of `self.target_mask` over matching references.
                This is a weighted sum. It is the numerator of `prob`.
            ``n_all``
                Unweighted number of matching reference objects.
            ``n_colors_used``
                Number of non-NaN test colors with non-NaN errors,
                including for objects skipped by `min_colors_used`.

        Notes
        -----
        Before the call to this function, feature selection is recommended.
        Especially when more than 3 photometric bands are used. See, for example,
        `select_features_grid`.

        ``n_good / n_all`` is generally not the probability:
        the denominator must be the sum of reference weights.

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
                raise ValueError("cpu_number must be a positive integer or None.")

        if (
            isinstance(block_size, (bool, np.bool_))
            or not isinstance(block_size, (int, np.integer))
            or block_size < 1
        ):
            raise ValueError("block_size must be a positive integer.")

        max_workers = max(mp.cpu_count() - 2, 1)

        n_jobs = (
            max_workers if cpu_number is None else min(int(cpu_number), max_workers)
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

        # Compile/load the Numba specialization before starting workers
        if self.N_test_objects > 0:
            self._matching_indices(0)

        def process_block(start):
            """Process a disjoint block and return its warning counts."""
            stop = min(start + block_size, self.N_test_objects)

            block_unmatched = 0
            block_single_match = 0
            block_below_min_colors = 0

            for i in range(start, stop):
                usable_i = ~np.isnan(self.test_dict["colors"][:, i]) & ~np.isnan(
                    self.test_dict["errors"][:, i]
                )
                n_colors_used[i] = int(usable_i.sum())

                if n_colors_used[i] < min_colors_used:
                    block_below_min_colors += 1
                    continue

                # The compiled kernel releases the GIL during matching
                indices = self._matching_indices(i)

                values = self.target_mask[indices]
                weights = self.reference_dict["weights"][indices]

                n = indices.size
                n_all[i] = n
                n_good[i] = values.sum()

                if n == 0:
                    block_unmatched += 1
                    continue

                # Dataset guarantees finite, strictly positive weights
                weight_sum = weights.sum()
                prob[i] = n_good[i] / weight_sum

                if n > 1:
                    residuals = (values - prob[i] * weights) / weight_sum

                    variance = n / (n - 1) * np.sum(residuals * residuals)
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

        block_starts = range(0, self.N_test_objects, block_size)

        # With n_jobs=1, use ordinary map without creating worker threads
        pool_context = (
            ThreadPoolExecutor(max_workers=int(n_jobs)) if n_jobs > 1 else nullcontext()
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

    def get_results_dataframe(self, inplace=False):
        """
        Return a copy of the test DataFrame with probability results.

        Preserve all original columns, the row order, and the index of
        the DataFrame contained in `test_dataset`, passed to the constructor.
        Append every entry of `self.probability_dictionary` as a column named
        ``fomoz_<key>``. Results are assigned by row position, so duplicate
        index labels are supported.

        When no explicit test dataset was supplied, return the reference
        sample used as the test dataset.

        Parameters
        ----------
        inplace : bool, default=False
            The test DataFrame is modified only when inplace=True.
            The returned object is the original DataFrame when inplace=True,
            or a separate copy otherwise.

        Returns
        -------
        pandas.DataFrame
            Copy or extension of the test DataFrame with the following added columns:

            fomoz_prob
                Weighted membership probability.
            fomoz_prob_error
                Approximate sampling standard error of the probability.
            fomoz_n_good
                Sum of weighted memberships over matching references,
                i.e. the numerator in the fomoz_prob expression.
            fomoz_n_all
                Unweighted number of matching reference objects.
            fomoz_n_colors_used
                Number of usable selected colors for each test object.

            Missing estimates are NaN.

        Raises
        ------
        RuntimeError
            If probability results are not available.

        Notes
        -----
        Do not modify or reorder the test DataFrame between `process`
        and this call. Results are associated by position; changes that
        preserve the number of rows cannot be detected by this method.

        Examples
        --------
        >>> point_prob.process()
        >>> results = point_prob.get_results_dataframe()
        >>> results[["fomoz_prob", "fomoz_prob_error"]]
        """
        if not self.probability_dictionary:
            raise RuntimeError(
                "No probability results are available. "
                "Run process() successfully before requesting the DataFrame."
            )

        test_frame = self.test_dataset.dataset
        result_columns = {}

        for key, values in self.probability_dictionary.items():
            column = f"fomoz_{key}"

            if column in test_frame.columns and not inplace:
                raise ValueError(
                    f"The test DataFrame already contains '{column}'. "
                    "Rename or remove the existing column before exporting."
                )

            values = np.asarray(values)

            if values.ndim != 1 or len(values) != len(test_frame):
                raise ValueError(
                    f"Result '{key}' must be a one-dimensional array "
                    f"with {len(test_frame)} entries; got shape {values.shape}."
                )

            result_columns[column] = values

        output = test_frame if inplace else test_frame.copy(deep=True)

        for column, values in result_columns.items():
            output[column] = values

        return output
