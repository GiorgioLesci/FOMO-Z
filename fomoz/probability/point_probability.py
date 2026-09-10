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
import numpy as np
from tqdm import tqdm

from ..auxiliary.statistics import compute_metrics, get_color_color_mask
from ..color_selection.dataset import Dataset
from ..color_selection.selection_definition import SelectionDefinition
from ..color_selection.calibration_grid import CalibrationGrid


class PointProbability:
    """Get point-estimate probabilities."""

    def __init__(
        self,
        reference_dataset: Dataset,
        test_dataset: Dataset,
        selection_definition: SelectionDefinition,
        verbose=True,
    ):
        r"""
        Estimate point-estimate probabilities from color grids.

        Given a reference dataset (defining the "true" color distribution, e.g. from
        a well-characterized sample) and a test dataset (the objects for
        which a probability is required), this class identifies, for each test object,
        which reference objects are color-consistent with it within its photometric
        color errors, and uses this information to estimate a point-estimate probability
        for the test object.

        The errors on magnitude measurements present in `test_dataset` are added in
        quadrature to obtain the corresponding color errors used for the color-matching.

        Parameters
        ----------
        reference_dataset : Dataset
            A `Dataset` object containing the reference dataset and band information.
        test_dataset : Dataset
            A `Dataset` object containing the test dataset and band information. It must
            include errors on magnitudes, which must be provided within the pandas DataFrame
            passed to the `Dataset` object. The keys representing magnitude error names must
            follow the convention `band_name + "_error"`. For example, if `mag_g` and `mag_r`
            are the band keys, the error keys must be `mag_g_error` and `mag_r_error`.
            These errors are added in quadrature to obtain the corresponding color errors.
        selection_definition : SelectionDefinition
            A `SelectionDefinition` object defining the targeted sample. If, for example, the user
            wants to get the point-estimate probabilities of test objects to lie above redshift
            0.5, then `selection_definition = SelectionDefinition(z_lim=0.5)`.
        verbose : bool, default=True
            If True, print messages during execution (e.g. referring to
            the time employed for computation, status of the calibration)

        Notes
        -----
        The methods implemented in this class are designed for estimating
        probabilities of the following kind:

        - probability of the test objects to lie above a given redshift;
        - probability of having a stellar mass below a given value;
        - probability of being a quasar.

        More generally, this class supports probabilities of exceeding/falling
        below a threshold, or of belonging to a discrete class, as defined by
        `selection_definition`. Due to the finite size of the reference sample,
        questions like "what is the probability of my test object to have a
        redshift exactly equal to a given value" are not well posed: with a finite
        reference sample, the probability of finding a reference object at any
        single, exact value is effectively zero, so such probabilities
        cannot be estimated reliably with this method. Only
        probabilities integrated over a finite region (e.g. above/below a
        threshold, or membership in a class) are meaningful.
        """
        if not isinstance(reference_dataset, Dataset):
            raise TypeError(
                f"'reference_dataset' must be a Dataset instance, got {type(reference_dataset).__name__}."
            )
        if not isinstance(test_dataset, Dataset):
            raise TypeError(
                f"'test_dataset' must be a Dataset instance, got {type(test_dataset).__name__}."
            )
        if not isinstance(selection_definition, SelectionDefinition):
            raise TypeError(
                f"'selection_definition' must be a SelectionDefinition instance, got {type(selection_definition).__name__}."
            )

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
                    "Different sets of colors in the reference and test datasets. The two lists of colors must match element by element."
                )
        for band in test_dataset.bands:
            if band + "_error" not in test_dataset.dataset.columns:
                raise ValueError(
                    f"Uncertainties on {band} are not defined. Include them in the '{band}_errors' column within the DataFrame of test_dataset."
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
        self.colors = None
        self.verbose = verbose

    def _set_dictionaries(self):
        r"""
        Build and store the color/error/weight matrices for reference and test data.

        Based on `self._build_color_dict`, this populates `self.reference_dict` and `self.test_dict`
        with the color matrices (and, respectively, weights and photometric color errors)
        computed from `self.reference_dataset` and `self.test_dataset`, using the colors
        currently stored in `self.colors` (either selected via feature selection, or
        set to all available colors as a fallback).

        This method must be called after `self.colors` has been set, and before any
        method relying on `self.reference_dict` or `self.test_dict` (e.g.
        `_color_difference`, `_mask`).
        """
        self.N_reference_objects = len(self.reference_dataset.dataset)
        self.N_test_objects = len(self.test_dataset.dataset)
        self.reference_dict = self._build_color_dict(
            self.reference_dataset.dataset, False, True
        )
        self.test_dict = self._build_color_dict(self.test_dataset.dataset, True, False)

    def _build_color_dict(self, dataset, include_errors, include_weights):
        r"""
        Build a matrix of colors (and optionally their errors and weights) from a Dataset.

        For each color in `self.colors`, the corresponding color array is computed as the
        difference between the two magnitude columns it refers to, and stored as a row
        in a `(n_colors, n_objects)` matrix. If `include_errors` is True, a matrix of
        the same shape is also built, with the propagated color errors obtained by
        summing the two magnitude errors in quadrature (missing magnitude errors
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

    def _color_difference(self, i):
        r"""
        Compute color differences between a test object and the reference sample.

        For the `i`-th test object, compute the absolute difference, in each color,
        between the test object's color and the color of every object in the
        reference sample.

        Parameters
        ----------
        i : int
            Index of the object in the test dataset for which the color differences
            are computed.

        Returns
        -------
        numpy.ndarray
            Array of shape `(n_colors, n_reference)` containing the absolute color
            differences, where `diff[k, j]` is the absolute difference, in the
            `k`-th color, between the `i`-th test object and the `j`-th reference
            object.
        """
        reference_colors = self.reference_dict["colors"]  # (n_colors, n_reference)
        test_color = self.test_dict["colors"][:, i : i + 1]  # (n_colors, 1)

        diff = np.abs(reference_colors - test_color)  # (n_colors, n_reference)
        return diff

    def _mask(self, i):
        r"""
        Select reference-dataset objects that are color-consistent with a test object.

        For the `i`-th object in the test dataset, this returns a boolean mask over
        the reference dataset, flagging the reference objects whose color, in every
        color defined for this class, differs from the test object's color by no more
        than the test object's photometric color error. A reference object is kept only
        if it satisfies this condition simultaneously in all colors.

        Parameters
        ----------
        i : int
            Index of the object in the test dataset for which the mask is computed.

        Returns
        -------
        numpy.ndarray of bool
            Boolean mask of length `self.N_reference_objects`, True for reference
            objects whose colors are compatible, within the test object's errors,
            with the `i`-th test object.
        """
        tol = self.test_dict["errors"][:, i : i + 1]  # (n_colors, 1)
        diff = self._color_difference(i)

        return np.all(diff <= tol, axis=0)  # (n_reference,)

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

    def process(self):
        r"""
        Compute point-estimate probabilities for all test objects.

        A selection of the informative features is recommended before the call
        to this function. See, for example, the `select_features_grid` method.
        If no feature selection has been performed (i.e. `self.colors` is not set),
        all colors available in `self.reference_dataset` are used, and a warning is
        raised.

        For each object in the test dataset, the reference objects that are
        color-consistent with it (see `_mask`) are identified, and the point-estimate
        probability for that test object is computed as the weighted mean of
        `self.target_mask` over these matched reference objects. Note that
        `self.target_mask` is not necessarily binary (0/1): it may contain
        intermediate values (e.g. derived from photometric redshift PDFs), in which
        case the resulting probability is a weighted average rather than a strict
        binomial proportion. See the documentation of the `SelectionDefinition`
        class for more details.

        The uncertainty on this probability is estimated as the standard error of
        the mean:

        .. math::
            \sigma_{P} = \frac{s}{\sqrt{n}}

        where `s` is the sample standard deviation (with Bessel's correction, i.e.
        dividing by `n - 1`) of the `self.target_mask` values among the matched
        reference objects, and `n` is the number of matched reference objects. This
        requires at least two matched reference objects (`n > 1`); for test objects
        with `n <= 1` matched reference objects, `"prob"` and/or `"prob_error"` are
        set to NaN (see Returns).

        Returns
        -------
        dict
            Dictionary with the following keys:

            - `"prob"` : numpy.ndarray of shape `(n_test,)`
              Point-estimate probability for each test object. NaN for test
              objects with no matched reference objects.
            - `"prob_error"` : numpy.ndarray of shape `(n_test,)`
              Standard error of the mean on `"prob"`, as described above. NaN
              for test objects with fewer than two matched reference objects.
            - `"n_good"` : numpy.ndarray of shape `(n_test,)`
              Sum of `self.target_mask` over the matched reference objects, for
              each test object.
            - `"n_all"` : numpy.ndarray of shape `(n_test,)`
              Number of matched reference objects, for each test object.
        """
        if self.colors is None:
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

        n_unmatched = 0
        n_single_match = 0

        for i in tqdm(
            range(self.N_test_objects),
            desc="Estimating point-estimate probabilities",
            disable=not self.verbose,
        ):
            # mask contains booleans
            # self.target_mask contains floats
            mask = self._mask(i)
            values = self.target_mask[mask]
            n_all[i] = np.count_nonzero(mask)
            n_good[i] = values.sum()

            if n_all[i] == 0:
                n_unmatched += 1
                continue

            prob[i] = n_good[i] / n_all[i]

            if n_all[i] > 1:
                prob_error[i] = values.std(ddof=1) / np.sqrt(n_all[i])
            else:
                n_single_match += 1

        if n_unmatched > 0:
            warnings.warn(
                f"{n_unmatched} test object(s) had no matching reference objects; "
                "their 'prob' and 'prob_error' were set to NaN."
            )
        if n_single_match > 0:
            warnings.warn(
                f"{n_single_match} test object(s) had only one matching reference "
                "object; their 'prob_error' was set to NaN."
            )

        return {
            "prob": prob,
            "prob_error": prob_error,
            "n_good": n_good,
            "n_all": n_all,
        }
