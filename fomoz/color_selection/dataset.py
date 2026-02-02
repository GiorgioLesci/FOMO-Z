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

    def __init__(self, dataset, bands, z_gal_string="redshift", missing_data=-99):
        """
        Set the data model for color selections.

        Class defining the color selection dataset.

        This class standardizes input photometric data and handles missing values.

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
            with NaN for consistency.
        """
        assert isinstance(
            dataset, pd.DataFrame
        ), "Input dataset must be a pandas DataFrame"

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
        self.colors = list(combinations(self.bands, 2))
        self.color_spaces = list(combinations(self.colors, 2))
