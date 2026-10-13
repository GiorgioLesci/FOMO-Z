"""
Reconstruct reference-property distributions from classification results.
"""

import numpy as np
import warnings

from ..auxiliary.distributions import get_distribution as _get_distribution
from ..binary_classification.linear_classification import LinearClassification
from ..binary_classification.grid_classification import GridClassification
from ..probability.point_probability import PointProbability


class Reconstruction:
    """
    Reconstruct property distributions from a classified reference sample.

    Binary classifications contribute through their selection masks.
    PointProbability contributes through its estimated membership
    probabilities, evaluated on the reference sample itself.

    Distributions use reference point estimates. Reference PDZs are
    not integrated when constructing the histograms.
    """

    def __init__(self, classification, zl_index=None):
        """
        Extract the reference sample and classification weights.

        Parameters
        ----------
        classification : `LinearClassification`, `GridClassification` or `PointProbability`
            Classification object whose process() method has already
            completed.

            For `PointProbability`, the test sample must be the reference
            Dataset itself: initialize it with test_dataset=None or
            explicitly pass the same `Dataset` as reference and test.
        zl_index : int, optional
            Index of the calibrated redshift threshold for a binary
            classification. Required when multiple redshift thresholds are
            available (i.e. it is required if `classification` is either a
            `LinearClassification` or a `GridClassification` object).
            Even if None, it defaults to zero when only one threshold is
            available.

            Must be `None` for `PointProbability`, which already represents
            a single selection definition.

        Notes
        -----
        This class does not run calibration, feature selection, or
        probability estimation.

        The reference `DataFrame` and prior weights are retained without
        making a full dataset copy. Do not modify or reorder the source
        dataset after classification.

        Create a new `Reconstruction` instance if the classification
        results change.

        Interpreting the reconstructed distribution as that of an
        independent test sample requires a representative reference
        sample.

        Reference objects with NaN membership probabilities are
        excluded from all reconstructed distributions, without
        modifying the source dataset. A warning reports their number.
        If all membership probabilities are NaN, a ValueError is raised.
        """
        if isinstance(classification, PointProbability):
            if zl_index is not None:
                raise ValueError("zl_index must be None for PointProbability.")

            if not classification.test_is_reference:
                raise ValueError(
                    "Reconstruction requires probabilities evaluated "
                    "on the reference sample itself. Initialize "
                    "PointProbability with test_dataset=None or with "
                    "the reference Dataset as test_dataset."
                )

            if "prob" not in classification.probability_dictionary:
                raise RuntimeError(
                    "Run PointProbability.process() successfully "
                    "before initializing Reconstruction."
                )

            reference = classification.reference_dataset

            self.dataset = reference.dataset
            self.prior_weights = self.dataset["weights"].to_numpy()
            self.classification_weights = np.asarray(
                classification.probability_dictionary["prob"]
            )
            self.missing_data = reference.missing_data
            self.zl_index = None

        elif isinstance(
            classification,
            (LinearClassification, GridClassification),
        ):
            n_thresholds = len(classification.td.z_lim)

            if zl_index is None:
                if n_thresholds != 1:
                    raise ValueError(
                        "Specify zl_index when the binary classification "
                        "contains multiple redshift thresholds."
                    )
                zl_index = 0

            if isinstance(zl_index, (bool, np.bool_)) or not isinstance(
                zl_index, (int, np.integer)
            ):
                raise TypeError("zl_index must be an integer.")

            if not 0 <= zl_index < n_thresholds:
                raise ValueError(
                    f"zl_index must lie between 0 and " f"{n_thresholds - 1}."
                )

            if isinstance(classification, LinearClassification):
                mask = classification.get_mask(
                    zl_index,
                    color_condition="all",
                )
            else:
                mask = classification.get_mask(
                    zl_index,
                    color_space="all",
                )

            self.dataset = classification.dataset
            self.prior_weights = np.asarray(classification.weights)
            self.classification_weights = np.asarray(mask)
            self.missing_data = classification.missing_data
            self.zl_index = int(zl_index)

        else:
            raise TypeError(
                "classification must be a LinearClassification, "
                "GridClassification, or PointProbability instance."
            )

        expected_shape = (len(self.dataset),)

        if (
            self.prior_weights.shape != expected_shape
            or self.classification_weights.shape != expected_shape
        ):
            raise ValueError(
                "Prior weights and classification results must have "
                "one entry per reference object."
            )

        # Exclude undefined PointProbability results from reconstruction
        # Binary classifications are left unchanged
        self._valid_probability = slice(None)

        if isinstance(classification, PointProbability):
            valid = ~np.isnan(self.classification_weights)
            n_missing = np.count_nonzero(~valid)

            if not np.any(valid):
                raise ValueError(
                    "Cannot reconstruct distributions: all membership "
                    "probabilities are NaN."
                )

            if n_missing:
                self._valid_probability = valid

                warnings.warn(
                    f"Excluding {n_missing} reference objects with "
                    "undefined membership probabilities from reconstruction. "
                    "Normalized distributions refer only to objects "
                    "with defined probabilities.",
                    stacklevel=2,
                )

    def get_distribution(self, variable, bin_edges, normalize=True):
        """
        Compute the distribution of a reference property.

        Parameters
        ----------
        variable : str
            Column name in the reference DataFrame, such as "redshift".
        bin_edges : array-like
            Finite, strictly increasing histogram bin edges.
        normalize : bool, default=True
            If True, return probability densities with unit integral
            over the supplied bin range. Otherwise, return weighted
            reference counts.

        Returns
        -------
        numpy.ndarray
            Histogram values, with one entry per bin.

        Raises
        ------
        KeyError
            If variable is absent from the reference DataFrame.
        ValueError
            If input values or weights are invalid, or a normalized
            histogram would have zero total weight.

        Notes
        -----
        Effective weights are the product of prior weights and
        classification weights. The latter are binary masks or
        membership probabilities, not target-definition weights.

        Objects with NaN membership probabilities from PointProbability
        are excluded, with a warning issued when Reconstruction is
        initialized. Normalized distributions refer to the remaining
        objects within the supplied bin range.

        Among retained objects, missing property values are allowed
        only for objects with zero effective weight. Infinite
        classification probabilities or values outside [0, 1]
        raise an error.

        For redshift, this method uses point estimates even when
        reference PDZs were supplied during classification.

        The reference sample is not restricted to objects that truly
        satisfy the target definition. Binary selections therefore
        retain their contaminants in the reconstructed distribution.

        Finite values outside the supplied bin range are excluded.
        Unnormalized counts describe the weighted reference sample;
        they are not absolute counts for an independent test sample.
        """
        if not isinstance(variable, str):
            raise TypeError("variable must be a column-name string.")

        if variable not in self.dataset.columns:
            raise KeyError(
                f"Column {variable!r} is not present in the " "reference dataset."
            )

        values = self.dataset[variable].to_numpy(
            dtype=float,
            na_value=np.nan,
        )

        return _get_distribution(
            values=values[self._valid_probability],
            bin_edges=bin_edges,
            prior_weights=self.prior_weights[self._valid_probability],
            classification_weights=self.classification_weights[
                self._valid_probability
            ],
            normalize=normalize,
            missing_data=self.missing_data,
        )
