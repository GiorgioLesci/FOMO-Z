"""
Input data handling for reference and test galaxy samples.

Store photometry, optional physical properties, object weights, and cumulative
redshift distributions, and define the colors available to FOMO-Z methods.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

from itertools import combinations
import re

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid


class Dataset:
    """
    Represent a reference or test sample for classification and reconstruction.

    The input table is copied, missing photometry is standardized, and color
    and color--color definitions are constructed from the supplied bands.
    Optional object weights and redshift probability density functions (PDZs)
    support weighted analyses and the propagation of reference redshift
    uncertainties.

    Attributes
    ----------
    dataset : pandas.DataFrame
        Internal table with a zero-based index. Contains the input columns,
        standardized ``"redshift"`` values, and a ``"weights"`` column.
    bands : list or tuple of str
        Photometric magnitude column names, in the supplied order.
    colors : list
        Ordered band pairs defining magnitude differences. For example,
        ``('g', 'r')`` denotes ``g - r``. These are definitions, not arrays
        of measured color values.
    color_spaces : list
        All distinct pairs of entries in ``colors``, defining the available
        two-dimensional color--color spaces.
    z_gal_string : str or None
        Input column designated for point-redshift selections, if any.
    missing_data : scalar or list
        Missing-value sentinels supplied to the constructor.
    CDZs : numpy.ndarray or None
        Normalized cumulative redshift distributions, with shape
        ``(n_objects, n_samples)`` and rows aligned with ``dataset``.
        Set to None when no PDZs are supplied.
    PDZ_x : numpy.ndarray or None
        Redshift grid of ``CDZs``, with shape ``(n_samples,)``.
        Set to None when no PDZs are supplied.
    """

    def __init__(
        self,
        dataset,
        bands,
        z_gal_string=None,
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
        Initialize a sample from a photometric catalog.

        Parameters
        ----------
        dataset : pandas.DataFrame
            One row per object, with a magnitude column for every entry in
            ``bands``. Additional properties used for classification or
            reconstruction are retained. The input table is not modified.

            Magnitude uncertainties, when used by probabilistic methods,
            must be stored in columns named ``<band>_error`` (for example,
            ``g_error``), in magnitude units. This constructor does not
            require, create, or validate these uncertainty columns.
        bands : list or tuple of str
            Names of the magnitude columns. Use distinct names starting with
            an ASCII letter or underscore and containing only ASCII letters,
            digits, or underscores. The lowercase characters ``c`` and ``s``
            are forbidden anywhere in a band name because they are reserved
            for color-cut parameters. Rename catalog columns if necessary.
        z_gal_string : str or None, default=None
            Column containing the point-estimate redshifts to use for redshift
            selections. Its values are copied to ``"redshift"``; the original
            column is retained. It must exist and cannot be a band name.
            If the input already contains ``"redshift"``, use that name or
            rename the existing column before designating another one.

            With None, no point-redshift column is designated for selections.
            An existing ``"redshift"`` column is preserved; otherwise one is
            added and filled with NaN. Reference PDZs remain available
            independently of this setting.
        PDZs : array-like of shape (n_objects, n_samples), optional
            Redshift probability densities sampled on the common ``PDZ_x``
            grid, in the same row order as ``dataset``. Values must be finite
            and non-negative, and each row must have a finite, strictly
            positive trapezoidal integral over the grid.

            Input normalization is not required: each PDZ is normalized over
            the supplied grid and cumulatively integrated using the
            trapezoidal rule. Only the resulting ``CDZs`` are stored; no
            ``PDZs`` attribute is created. Probability outside the supplied
            grid is not recovered by this normalization.
        PDZ_x : array-like of shape (n_samples,), optional
            Finite, strictly increasing redshift grid with at least two
            samples. Required when ``PDZs`` is supplied. Ignored otherwise.
        weights : array-like of shape (n_objects,), optional
            Finite, non-negative object weights, aligned with the input rows
            by position, not by pandas index labels. Reference weights set
            objects' relative contributions to calibration, probability
            estimation, or reconstruction. Test weights set their relative
            contributions to the reconstructed target population.

            Weights are stored without normalization in ``dataset["weights"]``.
            If omitted, all objects receive unit weight. Any existing
            ``"weights"`` column is overwritten; pass it explicitly to use it.
        missing_data : int, float, or list, optional
            Missing-value sentinels to replace with NaN in the selected band
            columns and in every column whose name contains ``"_error"``.
            Defaults to ``[99, -99, 999, -999, 99.9, -99.9, 999.9, -999.9]``.
            Existing NaNs do not need to be listed. Use an empty list if no
            additional sentinels are needed.

            No rows are removed during initialization. Sentinel values in
            other columns and numerical infinities are not automatically
            replaced. Use :meth:`drop_missing` to remove objects with invalid
            entries in explicitly chosen columns.
        custom_colors : list or tuple of band pairs, optional
            Colors to use instead of all pairwise combinations of ``bands``.
            Each pair is a two-element tuple or list of distinct band names,
            representing the first magnitude minus the second. All names
            must belong to ``bands``, and every supplied band must occur in
            at least one color. Repeated pairs, including reversed pairs
            such as ``('g', 'r')`` and ``('r', 'g')``, are rejected.
            The supplied color order and orientation are preserved.

            At least two colors are required, including when this object is
            used by a probabilistic method. With None, colors follow the
            order of ``bands`` and include each distinct band pair once.

        Raises
        ------
        TypeError
            If ``dataset`` is not a pandas DataFrame, ``bands`` is not a
            list or tuple of strings, or ``custom_colors`` is not a list
            or tuple.
        KeyError
            If a required magnitude column or the designated redshift
            column is absent.
        ValueError
            If band names, redshift-column settings, PDZs, weights, or
            custom colors fail validation, or fewer than two colors
            are defined.

        Notes
        -----
        For distinct input bands and no custom colors, the number of colors
        is ``N_band * (N_band - 1) / 2``. The number of available color--color
        spaces is ``N_color * (N_color - 1) / 2``. These spaces support binary
        selection calibration; probabilistic methods compare individual
        colors. Numerical color values are computed by the consuming methods,
        not stored as additional columns by this constructor.

        The probability helpers compute a color uncertainty by adding the
        corresponding magnitude uncertainties in quadrature, assuming
        uncorrelated errors between bands. The choice of whether reference
        uncertainties also enter the matching tolerance is made by the
        probabilistic method, not by ``Dataset``.

        For reference samples, cumulative PDZ differences provide probability
        mass in redshift-selection intervals or reconstructed redshift bins.
        Prior object weights and these redshift probabilities have distinct
        roles; PDZ normalization does not incorporate the prior weights.

        Examples
        --------
        Build a sample with three magnitude bands and point redshifts:

        >>> catalog = pd.DataFrame({
        ...     "g": [23.1, 24.0], "r": [22.7, 23.4], "i": [22.5, 23.0],
        ...     "g_error": [0.05, 0.10], "r_error": [0.04, 0.08],
        ...     "i_error": [0.04, 0.07], "z_spec": [0.4, 0.9],
        ... })
        >>> sample = Dataset(catalog, bands=["g", "r", "i"],
        ...                  z_gal_string="z_spec")
        >>> sample.colors
        [('g', 'r'), ('g', 'i'), ('r', 'i')]
        >>> len(sample.color_spaces)
        3

        Restrict the available colors while retaining all three bands:

        >>> sample = Dataset(catalog, bands=["g", "r", "i"],
        ...                  custom_colors=[("g", "r"), ("r", "i")])
        >>> sample.color_spaces
        [(('g', 'r'), ('r', 'i'))]
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
                total_int = trapezoid(pdzs, pdz_x, axis=1)

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

        # Preserve original columns and provide the standardized redshift key.
        if z_gal_string is not None:
            self.dataset["redshift"] = self.dataset[z_gal_string].copy()
        elif "redshift" not in self.dataset.columns:
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
        Validate user-defined photometric color pairs.

        Parameters
        ----------
        custom_colors : list or tuple of band pairs
            Each pair must contain two distinct string names from ``bands``.
            Every band must appear in at least one pair. Duplicate pairs
            are rejected regardless of orientation.

        Returns
        -------
        list
            Validated pairs in their supplied order and orientation.
            Individual pairs retain their original list or tuple type.

        Raises
        ------
        TypeError
            If the outer container is not a list or tuple.
        ValueError
            If a pair is malformed, contains an unknown or repeated band,
            duplicates another pair, or leaves a supplied band unused.

        Notes
        -----
        The constructor separately checks that at least two colors remain.
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
        Remove objects with invalid values in the specified columns in place.

        An object is removed if any selected column contains NaN, None,
        a pandas missing value, a sentinel in ``missing_data``, or positive
        or negative numerical infinity. Only the named columns determine
        which rows are retained.

        Parameters
        ----------
        columns : str or list or tuple of str
            Existing DataFrame columns to check. At least one non-empty
            column name is required. Repeated names are checked once.

        Returns
        -------
        None
            The internal table and any stored cumulative PDZs are updated
            in place. The number of removed objects is printed when nonzero.

        Raises
        ------
        TypeError
            If ``columns`` is not a string, list, or tuple, or contains
            entries other than non-empty strings.
        KeyError
            If a requested column does not exist.
        ValueError
            If no columns are specified, no valid objects would remain,
            or stored cumulative PDZs are not aligned with the table.
            The instance is not modified in these cases.

        Notes
        -----
        The retained rows keep their relative order and receive a new
        zero-based index. The same row mask is applied to ``CDZs``;
        ``PDZ_x`` is unchanged. Weights are retained without renormalization.
        The original input DataFrame is not modified.

        Examples
        --------
        >>> catalog = pd.DataFrame({"g": [23.1], "r": [22.7],
        ...                         "i": [22.5], "z_spec": [0.4]})
        >>> sample = Dataset(catalog, ["g", "r", "i"],
        ...                  z_gal_string="z_spec")
        >>> sample.drop_missing(["g", "r", "i", "redshift"])
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
