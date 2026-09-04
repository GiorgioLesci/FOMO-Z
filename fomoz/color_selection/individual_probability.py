"""
Derive probabilities for individual objects.

This module derives probabilities for individual objects based on color grids,
given any condition on redshift or any other galaxy properties. The reference
dataset defines a grid of objects with known colors, and for each test object,
a probability is estimated from the fraction of reference objects
that are color-consistent with it within its photometric errors.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import warnings
import numpy as np
from .dataset import Dataset


class IndividualProbability:
    """Get probabilities for individual objects."""

    def __init__(
        self,
        reference_dataset: Dataset,
        test_dataset: Dataset,
    ):
        r"""
        Estimate individual-object probabilities from color grids.

        Given a reference dataset (defining the "true" color distribution, e.g. from
        a well-characterized sample) and a test dataset (the objects for
        which a probability is required), this class identifies, for each test object,
        which reference objects are color-consistent with it within its photometric
        color errors, and uses this information to estimate an individual probability
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
        """
        if test_dataset.CDZs is not None:
            warnings.warn(
                "PDZs were set in test_dataset, but they are not used by the IndividualProbability class."
            )
        if not np.allclose(np.asarray(test_dataset.dataset["weights"].values), 1.0):
            warnings.warn(
                "Weights were set in test_dataset, but they are not used by the IndividualProbability class."
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

        self.N_reference_objects = len(reference_dataset.dataset)
        self.reference_dict = self._build_color_dict(
            reference_dataset.dataset, reference_dataset.colors, False, True
        )
        self.test_dict = self._build_color_dict(
            test_dataset.dataset, test_dataset.colors, True, False
        )
        self.colors = reference_dataset.colors

    def _build_color_dict(self, dataset, colors, include_errors, include_weights):
        r"""
        Build a matrix of colors (and optionally their errors and weights) from a Dataset.

        For each color in `colors`, the corresponding color array is computed as the
        difference between the two magnitude columns it refers to, and stored as a row
        in a `(n_colors, n_objects)` matrix. If `include_errors` is True, a matrix of
        the same shape is also built, with the propagated color errors obtained by
        summing the two magnitude errors in quadrature (missing magnitude errors
        propagate as NaN in the corresponding color error).

        Parameters
        ----------
        dataset : Dataset
            The `Dataset` object (reference or test) from which colors are computed.
        colors : list of tuple
            List of colors to compute, where each entry is a `(band1, band2)` tuple
            identifying the two magnitude columns to be subtracted (`band1 - band2`).
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
              Matrix of colors, one row per color in `colors`, one column per
              object in `dataset`.
            - `"errors"` : numpy.ndarray of shape `(n_colors, n_objects)`, optional
              Matrix of propagated color errors, same layout as `"colors"`.
              Only present if `include_errors` is True.
            - `"weights"` : numpy.ndarray of shape `(n_objects,)`, optional
              Per-object weights. Only present if `include_weights` is True.
        """
        n_colors = len(colors)
        n_obj = len(dataset)
        color_matrix = np.empty((n_colors, n_obj))

        if include_errors:
            error_matrix = np.empty((n_colors, n_obj))

        for k, color in enumerate(colors):
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
        test_color = self.test_dict["colors"][:, i:i+1]  # (n_colors, 1)

        diff = np.abs(reference_colors - test_color)  # (n_colors, n_reference)
        return diff

    def _rank_bins_by_population(self, i):
        r"""
        Rank the color bins of a test object by their population in the reference sample.

        For the `i`-th test object, each color defines a 1D bin: the interval of that
        color, centered on the test object's color value, with half-width equal to the
        test object's photometric error on that color. This method computes, for every
        color independently, the number of reference objects falling within that bin,
        and returns the colors sorted from the most to the least populated bin.

        Parameters
        ----------
        i : int
            Index of the object in the test dataset for which the bins are ranked.

        Returns
        -------
        order : numpy.ndarray of int
            Indices into `self.colors`, sorted by decreasing bin population (most populated
            color bin first).
        counts : numpy.ndarray of int
            Number of reference objects falling in each color bin, sorted in the same
            order as `order`.
        """
        tol = self.test_dict["errors"][:, i:i+1]  # (n_colors, 1)
        diff = self._color_difference(i)
        per_color_mask = diff <= tol  # (n_colors, n_reference)

        counts = per_color_mask.sum(axis=1)  # (n_colors,)
        order = np.argsort(counts)[::-1]  # decreasing

        return order, counts[order]

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
        tol = self.test_dict["errors"][:, i:i+1]  # (n_colors, 1)
        diff = self._color_difference(i)

        return np.all(diff <= tol, axis=0)  # (n_reference,)
