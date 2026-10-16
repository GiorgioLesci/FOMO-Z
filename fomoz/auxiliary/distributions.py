"""
Compute weighted distributions from point estimates or cumulative PDZs.

Combine prior weights with binary or probabilistic classification weights.
"""

import warnings

import numpy as np


def get_distribution(
    values,
    bin_edges,
    prior_weights,
    classification_weights,
    normalize=True,
    missing_data=None,
    *,
    CDZs=None,
    PDZ_x=None,
):
    """
    Compute a weighted distribution from point estimates or cumulative PDZs.

    Each object contributes its prior weight multiplied by its classification
    weight. With point estimates, this contribution enters a single bin.
    With cumulative PDZs, it is distributed according to the probability
    mass within each bin.

    Parameters
    ----------
    values : array-like of shape (n_objects,) or None
        Property point estimates. Required in point-estimate mode.
        Ignored when CDZs are supplied; pass None in that case.
    bin_edges : array-like of shape (n_bins + 1,)
        Finite, strictly increasing bin edges. For point estimates, bins
        are left-inclusive and right-exclusive, except the final bin,
        which includes its right edge.
    prior_weights : array-like of shape (n_objects,)
        Finite, non-negative prior weights.
    classification_weights : array-like of shape (n_objects,)
        Boolean selection mask or finite membership probabilities between
        zero and one. Must not already include the prior weights.
    normalize : bool, default=True
        Return a density normalized over the supplied range if True;
        otherwise return weighted bin counts.
    missing_data : scalar or sequence, optional
        Additional missing-value sentinels for point estimates.
        Nonfinite point estimates are always invalid.
        Ignored in PDZ mode.
    CDZs : array-like of shape (n_objects, n_grid), optional
        Cumulative redshift distributions aligned with the full input
        sample. Supplying CDZs activates PDZ mode and requires PDZ_x.
        Contributing rows must be finite, nondecreasing, and normalized
        from zero to one over the grid.
    PDZ_x : array-like of shape (n_grid,), optional
        Finite, strictly increasing redshift grid, with at least two points.
        Required together with CDZs.

    Returns
    -------
    histogram : numpy.ndarray of shape (n_bins,)
        Weighted counts, or probability densities if normalize is True.

    Raises
    ------
    TypeError
        If normalize is not boolean.
    ValueError
        If shapes, weights, edges, or contributing property values or
        cumulative distributions are invalid; or if normalization is
        requested without positive weight inside the supplied range.

    Notes
    -----
    Alignment is positional; pandas index labels are not used.
    Zero-weight objects may have missing property values or invalid CDF
    rows. Weights themselves must always be valid.

    Cumulative probabilities are linearly interpolated at bin edges.
    Below and above PDZ_x, cumulative values are zero and one, respectively.
    PDZs are assumed normalized on that grid; probability outside the
    original grid is unknown. PDZ mode reconstructs redshift only.

    Weight outside the bin range is excluded and reported with a warning.
    Normalized output therefore describes the distribution conditional
    on that range.

    Unnormalized output gives weighted reference counts, not the absolute
    population size of an independent test sample. No uncertainty on the
    resulting histogram is estimated.
    """
    edges = np.asarray(bin_edges, dtype=float)
    prior = np.asarray(prior_weights, dtype=float)
    classification = np.asarray(classification_weights, dtype=float)

    if prior.ndim != 1 or classification.shape != prior.shape:
        raise ValueError(
            "prior_weights and classification_weights must be "
            "one-dimensional arrays of the same length."
        )

    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.isfinite(edges).all()
        or not np.all(np.diff(edges) > 0)
    ):
        raise ValueError(
            "bin_edges must be finite, strictly increasing, "
            "and contain at least two edges."
        )

    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be a boolean.")

    if not np.isfinite(prior).all() or np.any(prior < 0):
        raise ValueError("prior_weights must contain finite, non-negative values.")

    if (
        not np.isfinite(classification).all()
        or np.any(classification < 0)
        or np.any(classification > 1)
    ):
        raise ValueError(
            "classification_weights must contain finite values " "between zero and one."
        )

    if (CDZs is None) != (PDZ_x is None):
        raise ValueError("CDZs and PDZ_x must be supplied together.")

    weights = prior * classification
    active = weights > 0
    total_weight = weights.sum()
    if not np.isfinite(total_weight):
        raise ValueError("Total effective weight must be finite.")

    if CDZs is None:
        if values is None:
            raise ValueError("Point-estimate mode requires values.")

        values = np.asarray(values, dtype=float)
        if values.shape != prior.shape:
            raise ValueError(
                "values must be one-dimensional and aligned with the weights."
            )

        invalid = ~np.isfinite(values)
        if missing_data is not None:
            invalid |= np.isin(values, missing_data)

        if np.any(active & invalid):
            raise ValueError(
                f"{np.count_nonzero(active & invalid)} contributing "
                "object(s) have missing or nonfinite property values."
            )

        histogram = np.histogram(
            values[active],
            bins=edges,
            weights=weights[active],
        )[0]

    else:
        grid = np.asarray(PDZ_x, dtype=float)
        cdfs = np.asarray(CDZs, dtype=float)

        if (
            grid.ndim != 1
            or grid.size < 2
            or not np.isfinite(grid).all()
            or not np.all(np.diff(grid) > 0)
        ):
            raise ValueError(
                "PDZ_x must be finite, strictly increasing, "
                "and contain at least two points."
            )

        if cdfs.shape != (prior.size, grid.size):
            raise ValueError("CDZs must align with the full sample and PDZ_x.")

        histogram = np.zeros(edges.size - 1)
        tolerance = 1e-10

        for index in np.flatnonzero(active):
            cdf = cdfs[index]

            if (
                not np.isfinite(cdf).all()
                or np.any(cdf < -tolerance)
                or np.any(cdf > 1 + tolerance)
                or np.any(np.diff(cdf) < -tolerance)
                or not np.isclose(cdf[0], 0, rtol=0, atol=tolerance)
                or not np.isclose(cdf[-1], 1, rtol=0, atol=tolerance)
            ):
                raise ValueError(
                    f"Contributing CDZ row {index} must be finite, "
                    "nondecreasing, and normalized from zero to one."
                )

            # Remove only numerical deviations accepted by the tolerance.
            cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
            cdf[0] = 0.0
            cdf[-1] = 1.0

            edge_cdf = np.interp(edges, grid, cdf, left=0.0, right=1.0)
            histogram += weights[index] * np.diff(edge_cdf)

    if not np.isfinite(histogram).all():
        raise ValueError("Weighted histogram contains nonfinite values.")

    retained_weight = histogram.sum()
    if not np.isfinite(retained_weight):
        raise ValueError("Weight within the supplied bins must be finite.")

    if total_weight > 0:
        lost_fraction = max(0.0, 1.0 - retained_weight / total_weight)
        if lost_fraction > 1e-12:
            warnings.warn(
                f"{lost_fraction:.6%} of effective weight lies outside "
                "the supplied bin range. Normalized output is "
                "conditional on the retained range.",
                stacklevel=2,
            )

    if normalize:
        if retained_weight <= 0:
            raise ValueError(
                "Cannot normalize: weight within the supplied bins " "must be positive."
            )
        histogram = histogram / retained_weight / np.diff(edges)

    return histogram
