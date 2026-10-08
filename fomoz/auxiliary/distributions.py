"""
Reconstruct distributions from reference point estimates.

Combine prior weights with binary or probabilistic classification
weights to compute weighted histograms of reference properties.
"""

import numpy as np


def get_distribution(
    values,
    bin_edges,
    prior_weights,
    classification_weights,
    normalize=True,
    missing_data=None,
):
    """
    Compute a weighted distribution of reference point estimates.

    Each object contributes the product of its prior weight and its
    classification weight to the bin containing its property value.

    Parameters
    ----------
    values : array-like of shape (n_objects,)
        Reference point estimates of the property to reconstruct.
        For redshift distributions, these are point-estimate redshifts:
        reference PDZs and CDZs are not integrated by this function.
    bin_edges : array-like of shape (n_bins + 1,)
        Finite, strictly increasing histogram bin edges.
        All bins are left-inclusive and right-exclusive, except the
        final bin, which includes its right edge.
    prior_weights : array-like of shape (n_objects,)
        Finite, non-negative prior weights.
    classification_weights : array-like of shape (n_objects,)
        Boolean selection mask or finite membership probabilities
        between zero and one. These must not already include the
        prior weights.
    normalize : bool, default=True
        If True, return a probability density normalized over the
        supplied bin range. Otherwise, return weighted bin counts.
    missing_data : scalar or sequence, optional
        Additional values denoting missing property estimates.
        NaN and infinite values are always considered invalid.

    Returns
    -------
    histogram : numpy.ndarray of shape (n_bins,)
        Weighted bin counts, or probability densities when normalized.
        For normalized output, the sum of histogram multiplied by
        the corresponding bin widths equals one.

    Raises
    ------
    ValueError
        If input shapes, weights, or bin edges are invalid; if a
        contributing object has an invalid property value; or if
        normalization is requested with zero total weight in the bins.

    Notes
    -----
    All object arrays must have the same row order. Alignment is
    positional; pandas index labels are not used.

    Objects with zero effective weight do not contribute and may
    have missing property values. Invalid classification weights,
    including NaN probabilities, raise an error rather than being
    silently treated as zero.

    Finite property values outside the bin range are excluded.
    Normalization therefore describes the distribution conditional
    on that range.

    Unnormalized output gives weighted reference counts, not an
    estimate of the absolute number of objects in an independent
    test sample.
    """
    values = np.asarray(values, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)
    prior_weights = np.asarray(prior_weights, dtype=float)
    classification_weights = np.asarray(
        classification_weights, dtype=float
    )

    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array.")

    if (
        prior_weights.shape != values.shape
        or classification_weights.shape != values.shape
    ):
        raise ValueError(
            "values, prior_weights, and classification_weights "
            "must be one-dimensional arrays of the same length."
        )

    if (
        bin_edges.ndim != 1
        or bin_edges.size < 2
        or not np.all(np.isfinite(bin_edges))
        or not np.all(np.diff(bin_edges) > 0)
    ):
        raise ValueError(
            "bin_edges must be finite, strictly increasing, "
            "and contain at least two edges."
        )

    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be a boolean.")

    if (
        not np.all(np.isfinite(prior_weights))
        or np.any(prior_weights < 0)
    ):
        raise ValueError(
            "prior_weights must contain finite, non-negative values."
        )

    if (
        not np.all(np.isfinite(classification_weights))
        or np.any(classification_weights < 0)
        or np.any(classification_weights > 1)
    ):
        raise ValueError(
            "classification_weights must contain finite values "
            "between zero and one."
        )

    effective_weights = prior_weights * classification_weights
    contributing = effective_weights > 0

    invalid_values = ~np.isfinite(values)

    if missing_data is not None:
        invalid_values |= np.isin(values, missing_data)

    invalid_contributors = contributing & invalid_values
    if np.any(invalid_contributors):
        raise ValueError(
            f"{np.count_nonzero(invalid_contributors)} contributing "
            "object(s) have missing or infinite property values. "
            "Clean the reference property with Dataset.drop_missing() "
            "before initializing calibration or probability objects."
        )

    histogram, _ = np.histogram(
        values[contributing],
        bins=bin_edges,
        weights=effective_weights[contributing],
        density=False,
    )

    if not np.all(np.isfinite(histogram)):
        raise ValueError("Weighted histogram contains nonfinite values.")

    if normalize:
        total_weight = histogram.sum()

        if not np.isfinite(total_weight) or total_weight <= 0:
            raise ValueError(
                "Cannot normalize the distribution: total weight "
                "within the supplied bins must be finite and positive."
            )

        histogram = histogram / total_weight / np.diff(bin_edges)

    return histogram
