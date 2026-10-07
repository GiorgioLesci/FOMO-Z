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
            If `PDZs` is `None`, it is used in the definition of the targeted sample and
            in metrics computation. It is always used to derive the redshift distributions
            by the `get_redshift_distribution` methods of the calibration classes.
        PDZs : array-like of array-like, default=None
            Redshift probability functions (normalization is not required) used
            in the metrics computation. If not provided, point-estimates are used
            (whose key is defined by `z_gal_string`). Must have shape `(len(dataset), len(PDZ_x))`.
        PDZ_x : array-like, default=None
            Redshift values where `PDZs[i]` is computed.
        weights : array-like, default=None
            Prior per-object weights used to compute the selection metrics.
        missing_data : int, float, or list, optional
            Value(s) used in the dataset to denote missing entries. These are replaced
            with NaN for consistency (if missing data in your data set are NaN,
            you don't need to specify it). This constructor replaces with NaN only
            the missing values of the photometric bands. To remove (not replace)
            missing data in other columns of the `dataset` `DataFrame` use
            `Dataset.drop_missing()`.
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
            errMsg = f"z_gal_string='{z_gal_string}' not found in dataset columns. "
            errMsg += "If redshift is not needed, set z_gal_string=None."
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

        # Validate and normalize reference PDZs
        self.CDZs = None
        self.PDZ_x = None

        if PDZs is not None:
            if PDZ_x is None:
                raise ValueError("PDZ_x must be provided when PDZs are supplied.")

            try:
                pdzs = np.asarray(PDZs, dtype=float)
                pdz_x = np.asarray(PDZ_x, dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "PDZs and PDZ_x must contain numerical values."
                ) from error

            if pdzs.ndim != 2:
                raise ValueError(
                    "PDZs must be a two-dimensional array with shape "
                    "(n_objects, n_samples)."
                )

            if pdz_x.ndim != 1 or pdz_x.size < 2:
                raise ValueError(
                    "PDZ_x must be a one-dimensional array "
                    "containing at least two samples."
                )

            if not np.all(np.isfinite(pdz_x)):
                raise ValueError("PDZ_x must contain only finite values.")

            if not np.all(np.diff(pdz_x) > 0):
                raise ValueError("PDZ_x must be strictly increasing.")

            if pdzs.shape != (len(dataset), pdz_x.size):
                raise ValueError(
                    f"PDZs must have shape ({len(dataset)}, {pdz_x.size}); "
                    f"got {pdzs.shape}."
                )

            invalid_rows = np.any(
                ~np.isfinite(pdzs) | (pdzs < 0),
                axis=1,
            )
            if np.any(invalid_rows):
                raise ValueError(
                    f"{np.count_nonzero(invalid_rows)} PDZ(s) contain "
                    "nonfinite or negative densities. "
                    "PDZ values must be finite and non-negative."
                )

            # Individual density samples may be zero, but each PDZ
            # must have a finite, strictly positive integral.
            with np.errstate(over="ignore", invalid="ignore"):
                total_int = np.trapz(pdzs, pdz_x, axis=1)

            invalid_integrals = ~np.isfinite(total_int) | (total_int <= 0)
            if np.any(invalid_integrals):
                raise ValueError(
                    f"{np.count_nonzero(invalid_integrals)} PDZ(s) have "
                    "a nonfinite or non-positive integral over PDZ_x. "
                    "Each PDZ must have a finite, strictly positive integral."
                )

            with np.errstate(over="ignore", invalid="ignore"):
                pdzs_norm = pdzs / total_int[:, None]

            if not np.all(np.isfinite(pdzs_norm)):
                raise ValueError(
                    "PDZ normalization produced nonfinite densities. "
                    "Check the density values and the PDZ_x sampling grid."
                )

            dz = np.diff(pdz_x)
            cdf = np.zeros_like(pdzs_norm)
            cdf[:, 1:] = np.cumsum(
                (0.5 * pdzs_norm[:, :-1] + 0.5 * pdzs_norm[:, 1:]) * dz,
                axis=1,
            )

            self.PDZ_x = pdz_x
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

        # Check the redshifts
        if "redshift" not in self.dataset.columns and z_gal_string is not None:
            self.dataset["redshift"] = self.dataset[z_gal_string].values
            self.dataset = self.dataset.drop(columns=[z_gal_string])

        if z_gal_string is None:
            self.dataset["redshift"] = np.nan

        self.z_gal_string = z_gal_string
        self.missing_data = missing_data

        # Prior weights
        if weights is None:
            self.dataset["weights"] = np.ones(len(self.dataset))
        else:
            weights = np.asarray(weights, dtype=float)

            if weights.shape != (len(self.dataset),):
                raise ValueError(
                    "weights must be a one-dimensional array "
                    "with one entry per object."
                )

            if np.any(~np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("weights must contain non-negative and finite values.")

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

    def drop_missing(self, columns):
        """
        Remove objects with missing or infinite values in selected columns.

        Invalid entries include NaN, None, pandas missing values,
        sentinels declared in ``self.missing_data``, and positive or
        negative numerical infinity. An object is removed if any of
        the specified columns contains an invalid entry.

        Parameters
        ----------
        columns : str or sequence of str
            DataFrame column names to check. Only these columns
            determine which objects are removed.

        Notes
        -----
        This method modifies the Dataset in place and resets its
        DataFrame index.
        The same row mask is applied to the stored cumulative redshift
        distributions, while their sampling grid remains unchanged.
        """
        if isinstance(columns, str):
            columns = [columns]
        elif isinstance(columns, (list, tuple)):
            columns = list(columns)
        else:
            raise TypeError("columns must be a string or a list/tuple of strings.")

        if not columns:
            raise ValueError("Specify at least one column.")

        if any(not isinstance(column, str) or not column.strip() for column in columns):
            raise TypeError("Column names must be non-empty strings.")

        columns = list(dict.fromkeys(columns))

        unknown = [column for column in columns if column not in self.dataset.columns]
        if unknown:
            raise KeyError(f"Columns not found in dataset: {unknown}")

        sentinels = self.missing_data
        if sentinels is None:
            sentinels = []
        elif np.isscalar(sentinels):
            sentinels = [sentinels]
        else:
            sentinels = list(sentinels)

        invalid = np.zeros(len(self.dataset), dtype=bool)

        for column in columns:
            values = self.dataset[column]
            invalid |= (
                values.isna() | values.isin(sentinels) | values.isin([np.inf, -np.inf])
            ).to_numpy(dtype=bool)

        keep = ~invalid
        if not np.any(keep):
            raise ValueError(
                f"No objects have valid values in all requested "
                f"columns: {columns}. Dataset was not modified."
            )

        cdzs = None
        if self.CDZs is not None:
            cdzs = np.asarray(self.CDZs)
            if cdzs.ndim != 2 or cdzs.shape[0] != len(self.dataset):
                raise ValueError(
                    "Stored CDZs are not aligned with the DataFrame. "
                    "Dataset was not modified."
                )

        n_removed = int(np.count_nonzero(invalid))
        if n_removed == 0:
            self.dataset.reset_index(drop=True, inplace=True)
            return

        # Prepare both filtered structures before updating the instance.
        filtered_dataset = self.dataset.loc[keep].reset_index(drop=True)
        filtered_cdzs = cdzs[keep] if cdzs is not None else None

        self.dataset = filtered_dataset
        self.CDZs = filtered_cdzs

        print(f"The Dataset.drop_missing() method removed {n_removed} objects.")
