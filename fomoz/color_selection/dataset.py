"""
Dataset settings.

This module handles dataset properties.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

from itertools import combinations

import numpy as np
import pandas as pd


class Dataset:
    """Set the data model for the derivation of color selections."""

    def __init__(
        self,
        dataset,
        bands,
        z_gal_string="redshift",
        missing_data=-99,
        custom_colors=None,
    ):
        """
        Set the data model for color selections.

        Class defining the color selection dataset.

        This class standardizes input photometric data and handles missing values.
        If `custom_colors` is not provided, all possible colors based on the input
        set of bands are considered.

        Parameters
        ----------
        dataset : pandas.DataFrame
            Input photometric dataset containing magnitude measurements and a
            redshift column.
        bands : list of str
            Names of the photometric bands to include in the analysis.
        z_gal_string : str, default='redshift'
            Column name in `dataset` that contains the galaxy redshift values.
        missing_data : int or float, default=-99
            Value used in the dataset to denote missing entries. These are replaced
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

        if z_gal_string not in dataset.columns:
            errMsg = f"z_gal_string='{z_gal_string}' "
            errMsg += "not found in dataset columns."
            raise KeyError(errMsg)

        if "c" in bands or "s" in bands:
            raise ValueError("Band keys cannot equal 'c' or 's'.")

        if len(bands) < 2:
            raise ValueError("At least two photometric bands are required.")

        if z_gal_string in bands:
            errMsg = f"z_gal_string='{z_gal_string}' conflicts "
            errMsg += f"with a band name in {bands}."
            raise ValueError(errMsg)

        if "redshift" in dataset.columns and z_gal_string != "redshift":
            errMsg = "Dataset already contains a 'redshift' column. "
            errMsg += "Please rename it or set z_gal_string='redshift'."
            raise ValueError(errMsg)

        needed_columns = [z_gal_string] + bands
        self.dataset = dataset[needed_columns].copy()
        for band in bands:
            self.dataset[band] = self.dataset[band].replace(missing_data, np.nan)
        self.dataset = self.dataset.rename(columns={z_gal_string: "redshift"})
        self.dataset["redshift"] = self.dataset["redshift"].replace(
            missing_data, np.nan
        )

        self.bands = bands

        if custom_colors is None:
            self.colors = list(combinations(self.bands, 2))
        else:
            self.colors = self._validate_custom_colors(custom_colors)

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
