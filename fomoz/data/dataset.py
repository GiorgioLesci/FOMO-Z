"""
Dataset settings.

This module handles dataset properties.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

from itertools import combinations
import re

import numpy as np
import pandas as pd


class Dataset:
    """Set the data model for the derivation of color selections."""

    def __init__(
        self,
        dataset,
        bands,
        z_gal_string="redshift",
        PDZs=None,
        PDZ_x=None,
        weights=None,
        missing_data=[
            99,
            -99,
            999,
            -999,
            99.9,
            -99.9,
            999.9,
            -999.9,
        ],
        custom_colors=None,
    ):
        """
        Set the data model for color selections.

        Class defining the color selection dataset.

        This class standardizes input photometric data and handles missing values.
        If `custom_colors` is not provided, all possible colors based on the input
        set of bands are considered. In the calibrations, objects can be weighted
        based on prior weights and/or photo-z probability density functions.
        These weighting schemes increase the execution time.

        Parameters
        ----------
        dataset : pandas.DataFrame
            Input photometric dataset containing magnitude measurements and,
            optionally, estimates of redshift and other physical properties.
        bands : list of str
            Names of the photometric bands to include in the analysis.
        z_gal_string : str, default="redshift"
            Column name in `dataset` that contains the point-estimate redshift values.
            If redshift information is not needed, set it to `None`.
            It is used in the metrics computation only if `PDZs` is `None`. It is always
            used to derive the redshift distributions by the `get_redshift_distribution`
            methods of the calibration classes.
        PDZs : array-like of array-like, default=None
            Redshift probability functions (normalization is not required) used
            in the metrics computation. If not provided, point-estimates are used
            (whose key is defined by `z_gal_string`). Must have shape `(len(dataset), len(PDZ_x))`.
        PDZ_x : array-like, default=None
            Redshift values where `PDZs[i]` is computed.
        weights : array-like, default=None
            Prior per-object weights used to compute the selection metrics.
        missing_data : int, float, or list, default=[99, -99]
            Value(s) used in the dataset to denote missing entries. These are replaced
            with NaN for consistency (if missing data in your data set are NaN,
            you don't need to specify it).
        custom_colors : list of tuple(str, str), optional
            List of colors to use instead of all pairwise combinations of the input `bands`.
            Each color must be specified as a 2-element tuple of band names,
            representing the difference between the first and second band
            (e.g., ('g', 'r') for the color g−r).
            If provided, `custom_colors` must satisfy the following conditions:

            - All band names must be drawn from the `bands` parameter.
            - Each band listed in `bands` must appear in at least one color.
            - Each color must consist of two distinct bands.
            - Duplicate colors are not allowed (e.g., ('g', 'r') and ('r', 'g')).

            If `custom_colors` is None (default), the full set of unique
            pairwise band combinations is used.

            Notes
            -----
            The choice of `custom_colors` directly determines the set of
            color–color spaces (`self.color_spaces`), which are constructed
            as all pairwise combinations of the selected colors.
        """
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas.DataFrame")

        if not isinstance(bands, (list, tuple)):
            raise TypeError("bands must be a sequence of strings")

        if not all(isinstance(b, str) for b in bands):
            raise TypeError("bands must contain only strings")

        if any("c" in b or "s" in b for b in bands):
            raise ValueError("Band keys cannot contain the characters 'c' or 's'.")

        if z_gal_string is not None and z_gal_string not in dataset.columns:
            errMsg = f"z_gal_string='{z_gal_string}' "
            errMsg += "not found in dataset columns."
            raise KeyError(errMsg)

        if z_gal_string in bands:
            errMsg = f"z_gal_string='{z_gal_string}' conflicts "
            errMsg += f"with a band name in {bands}."
            raise ValueError(errMsg)

        if "redshift" in dataset.columns and (
            z_gal_string != "redshift" and z_gal_string is not None
        ):
            errMsg = "Dataset already contains a 'redshift' column. "
            errMsg += "Please rename it or set z_gal_string='redshift'."
            raise ValueError(errMsg)

        # PDZs
        self.CDZs = None
        self.PDZ_x = np.asarray(PDZ_x)

        if PDZs is not None:

            # Check the input values
            if np.asarray(PDZs).shape[0] != len(dataset):
                raise ValueError("PDZs.shape[0] != len(dataset)")
            if np.asarray(PDZs).shape[1] != len(PDZ_x):
                raise ValueError("PDZs.shape[1] != len(PDZ_x)")
            if not np.all(np.diff(PDZ_x) > 0):
                raise ValueError("PDZ_x must be strictly increasing (sorted).")

            # Normalize
            total_int = np.trapz(np.asarray(PDZs), np.asarray(PDZ_x), axis=1)
            PDZs_norm = np.asarray(PDZs) / total_int[:, None]

            dz = np.diff(np.asarray(PDZ_x))
            cdf = np.zeros_like(PDZs_norm)
            cdf[:, 1:] = np.cumsum(
                0.5 * (PDZs_norm[:, :-1] + PDZs_norm[:, 1:]) * dz, axis=1
            )
            self.CDZs = cdf

        # Check if band keys have allowed names
        allowed_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        for band in bands:
            if not allowed_pattern.fullmatch(band):
                raise ValueError(
                    f"Invalid band name '{band}'. Must start with letter/underscore and contain only letters, numbers, underscores."
                )

        # Make a copy of the dataset
        self.dataset = dataset.copy().reset_index(drop=True)

        # Check for missing data - photometric bands
        self.dataset[bands] = self.dataset[bands].replace(missing_data, np.nan)

        error_columns = [col for col in self.dataset.columns if "_error" in col]
        self.dataset[error_columns] = self.dataset[error_columns].replace(
            missing_data, np.nan
        )

        # Check for missing data - redshifts
        if "redshift" not in self.dataset.columns and z_gal_string is not None:
            self.dataset["redshift"] = self.dataset[z_gal_string].values
            self.dataset = self.dataset.drop(columns=[z_gal_string])

        if z_gal_string is None:
            self.dataset["redshift"] = np.nan

        self.dataset["redshift"] = self.dataset["redshift"].replace(
            missing_data, np.nan
        )

        # Prior weights
        if weights is None:
            self.dataset["weights"] = np.ones(len(self.dataset))
        else:
            if np.any(np.asarray(weights) < 0) or np.any(np.isinf(np.asarray(weights))):
                raise ValueError("weights contains negative or infinite values.")
            self.dataset["weights"] = weights

        # Set bands, colors, and color spaces
        self.bands = bands

        if custom_colors is None:
            self.colors = list(combinations(self.bands, 2))
        else:
            self.colors = self._validate_custom_colors(custom_colors)

        if len(self.colors) < 2:
            raise ValueError(
                f"At least two colors are required (current number of colors: {len(self.colors)})."
            )

        self.color_spaces = list(combinations(self.colors, 2))

    def _validate_custom_colors(self, custom_colors):
        """
        Validate and standardize a user-defined list of photometric colors.

        Ensures that each color is a tuple of two distinct band names drawn
        from `self.bands`, that all bands appear in at least one color, and
        that no duplicate or invalid colors are provided.

        Parameters
        ----------
        custom_colors : list of tuple(str, str)
            User-defined photometric colors.

        Returns
        -------
        list of tuple(str, str)
            Validated list of user-defined colors (custom_colors).
        """
        if not isinstance(custom_colors, (list, tuple)):
            raise TypeError("custom_colors must be a list of (band1, band2) tuples.")

        bands_set = set(self.bands)
        used_bands = set()
        validated_colors = []
        seen_colors = set()  # order-insensitive

        for color in custom_colors:
            if (
                not isinstance(color, (list, tuple))
                or len(color) != 2
                or not all(isinstance(b, str) for b in color)
            ):
                raise ValueError(
                    "Each custom color must be a tuple of two band names, e.g. ('g', 'r')."
                )

            b1, b2 = color

            if b1 == b2:
                raise ValueError(f"Invalid color {color}: bands must be different.")

            if b1 not in bands_set or b2 not in bands_set:
                raise ValueError(
                    f"Invalid color {color}: all bands must be in bands={self.bands}."
                )

            key = frozenset((b1, b2))
            if key in seen_colors:
                raise ValueError(
                    f"Duplicate color detected (order-insensitive): {b1}-{b2}"
                )

            seen_colors.add(key)
            validated_colors.append(color)
            used_bands.update(color)

        missing = bands_set - used_bands
        if missing:
            raise ValueError(
                f"custom_colors does not include all bands. Missing: {sorted(missing)}"
            )

        return validated_colors
