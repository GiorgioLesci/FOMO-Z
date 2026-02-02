"""
Object selection class.

This class selects objects in an input photometric sample,
applying a calibrated color selection for any redshift threshold.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import copy
import os
import re

import numexpr as ne
import numpy as np
import yaml
from scipy.interpolate import BSpline

from .tools import set_input_variable_precision


class SelectObject:
    """Apply precomputed optimal color selections to datasets."""

    def __init__(self):
        """
        Class to apply precomputed optimal color selections to datasets.

        This class manages dictionaries of optimal cut parameters (either loaded
        directly or from file), reconstructs their interpolators,
        and applies the cuts to input datasets across a redshift thresholds.
        The output is a set of boolean masks marking which objects satisfy the cuts
        at each redshift threshold.

        Attributes
        ----------
        dict : dict
            Dictionary containing the optimal cut parameters and their associated
            interpolators. Keys are expressions (strings) to be evaluated via
            `numexpr`. The ``"metadata"`` key stores the metadata.
        mask : list of numpy.ndarray
            Boolean masks for each redshift in the grid, computed after calling
            ``process``.
        """
        self.dict = {}
        self.mask = []

    def _reconstruct_interpolators(self):
        """
        Reconstruct interpolator objects.

        This method reconstructs the
        interpolator functions for each cut expression.

        After reconstruction, the `self.dict` entries will also include:
          - ``c_interp`` : `scipy.interpolate.BSpline`
              The interpolator object for the intercept parameter.
          - ``s_interp`` : `scipy.interpolate.BSpline`, optional
              The interpolator object for the slope parameter (if present).

        Returns
        -------
        None
            Updates ``self.dict`` in place with the loaded dictionary and
            reconstructed interpolator objects.
        """
        bands = self.dict["metadata"]["bands"]
        if "c" in bands or "s" in bands:
            raise ValueError("Band keys cannot equal 'c' or 's'.")

        for key, entry in self.dict.items():

            # skip metadata
            if key == "metadata":
                continue

            if "knots" not in entry:
                raise ValueError(f"Expression '{key}' missing 'knots'")
            if "c_coefficients" not in entry:
                raise ValueError(f"Expression '{key}' missing 'c_coefficients'")
            if "s*" in key and "s_coefficients" not in entry:
                raise ValueError(f"Expression '{key}' missing 's_coefficients'")

            entry["knots"] = np.array(entry["knots"], dtype=np.float32).round(6)

            # reconstruct c interpolator if present
            if "order" in entry:

                entry["c_interp"] = BSpline(
                    entry["knots"],
                    entry["c_coefficients"],
                    entry["order"],
                    extrapolate=False,
                )

                # reconstruct s interpolator if present
                if "s_coefficients" in entry:
                    entry["s_interp"] = BSpline(
                        entry["knots"],
                        entry["s_coefficients"],
                        entry["order"],
                        extrapolate=False,
                    )

    def _select_objects(self, z_lim, mags):
        """
        Select based on linear color cuts.

        Select objects at a given redshift threshold.
        - If calibrated limiting redshifts are lower limits:
          - if `z_lim` is lower than the minimum limiting redshift, `z_min`,
            the selection is applied as if `z_lim` equals `z_min`;
          - if `z_lim` is larger than the maximum limiting redshift, `z_max`,
            no selection is applied.
        - If calibrated limiting redshifts are upper limits:
          - if `z_lim` > `z_max`, then `z_lim` = `z_max` is imposed;
          - if `z_lim` < `z_min`, then no selection is applied.

        For each cut expression in ``self.dict``, this method:
        1. Interpolates (or uses fixed) cut parameters ``c`` and ``s``
           at the given redshift.
        2. Evaluates the corresponding logical expression with the
           magnitudes in ``mags`` using `numexpr`.
        3. Combines results across all expressions with a logical OR.

        Parameters
        ----------
        z_lim : float
            Redshift threshold at which the cuts should be evaluated.
        mags : dict
            Dictionary mapping magnitude names
            to numpy arrays of magnitudes for all objects.

        Returns
        -------
        numpy.ndarray of bool
            Boolean mask array of shape (n_objects,), where True indicates
            that the object satisfies at least one of the color conditions.
        """
        mask = np.full(len(mags[list(mags.keys())[0]]), False)
        local_dict = dict(mags)
        local_dict["c"] = None
        local_dict["s"] = None
        for expr, expr_data in self.dict.items():

            # skip meta data
            if expr == "metadata":
                continue

            # adjust z_l
            z_l_points = expr_data["knots"]

            z_l = z_lim
            if self.dict["metadata"]["z_l_is_lower_limit"] and z_lim < min(z_l_points):
                z_l = min(z_l_points)
            elif not self.dict["metadata"]["z_l_is_lower_limit"] and z_lim > max(
                z_l_points
            ):
                z_l = max(z_l_points)

            # apply the selection
            condition = (
                z_l <= max(z_l_points)
                if self.dict["metadata"]["z_l_is_lower_limit"]
                else z_l >= min(z_l_points)
            )
            if condition:
                # set c
                if "c_interp" in expr_data:
                    local_dict["c"] = expr_data["c_interp"](z_l)
                else:
                    local_dict["c"] = expr_data["c_coefficients"]

                # set s
                if "s_interp" in expr_data:
                    local_dict["s"] = expr_data["s_interp"](z_l)
                elif "s_coefficients" in expr_data:
                    local_dict["s"] = expr_data["s_coefficients"]

                # evaluate the mask
                mask |= ne.evaluate(expr, local_dict=local_dict)

        return mask

    def process(self, z_lim, dataset, missing_data=-99):
        """
        Apply optimal color selections to a dataset.

        For each redshift in ``z_lim``, this method evaluates the stored
        color conditions based on the magnitudes in ``dataset`` and produces
        a boolean mask. All results are stored in ``self.mask``.
        - If calibrated limiting redshifts are lower limits:
          - if `z_lim` is lower than the minimum limiting redshift, `z_min`,
            the selection is applied as if `z_lim` equals `z_min`;
          - if `z_lim` is larger than the maximum limiting redshift, `z_max`,
            no selection is applied.
        - If calibrated limiting redshifts are upper limits:
          - if `z_lim` > `z_max`, then `z_lim` = `z_max` is imposed;
          - if `z_lim` < `z_min`, then no selection is applied.

        Parameters
        ----------
        z_lim : array-like or float
            Redshift threshold(s) at which the cuts are applied.
        dataset : pandas.DataFrame
            Table containing the photometric dataset. Must have columns
            corresponding to the magnitude names stored in
            ``self.dict['metadata']['bands']``.
        missing_data : int or float, default=-99
            Value used in the dataset to denote missing entries. These are replaced
            with NaN for consistency.

        Returns
        -------
        None
            Updates ``self.mask`` with a list of boolean arrays, one for each
            redshift in ``z_lim``.
        """
        # First checks
        if not self.dict:
            raise ValueError(
                "A color cut dictionary must be set",
                "before the call of this function.",
            )

        self.z_lim = set_input_variable_precision(z_lim)

        required_bands = self.dict["metadata"]["bands"]
        missing_bands = [band for band in required_bands if band not in dataset.columns]
        if missing_bands:
            raise ValueError(
                f"Dataset is missing required bands: {missing_bands}. "
                f"Available columns: {list(dataset.columns)}. If you want "
                "to use this selection without a given band, create columns "
                "in the source dataset for each missing band. Fill such columns "
                "with magnitude missing values (see the documentation of this method)."
            )

        # Handle missing data
        for band in required_bands:
            dataset[band] = dataset[band].replace(missing_data, np.nan)

        # Magnitude mapping
        mags = {
            band: dataset[band].to_numpy() for band in self.dict["metadata"]["bands"]
        }

        # Apply the selection
        self.mask = []
        for z_l in self.z_lim:
            self.mask.append(self._select_objects(z_l, mags))

    def load_parameters(self, calibration):
        """
        Load selection parameters from a ``CalibrateLinear`` instance.

        Load precomputed optimal color cut parameters
        and reconstruct interpolator objects.

        Parameters
        ----------
        calibration : CalibrateLinear
            `CalibrateLinear` object.

        Returns
        -------
        None
            Updates ``self.dict`` in place with the loaded dictionary and
            reconstructed interpolator objects.
        """
        dictionary = calibration.dict
        self.dict = copy.deepcopy(dictionary)
        self._reconstruct_interpolators()

    def load_parameters_from_file(self, input_dir, file_name):
        """
        Load selection parameters from file.

        Load precomputed optimal color cut parameters from a YAML file
        and reconstruct interpolator objects.

        This method reads a YAML-formatted dictionary of optimal selection cuts
        from file.

        Parameters
        ----------
        input_dir : str
            Directory where the YAML file is stored.
        file_name : str
            Name of the file to load (must include extension, e.g. "cuts.yaml").

        Returns
        -------
        None
            Updates ``self.dict`` in place with the loaded dictionary and
            reconstructed interpolator objects.
        """
        file_path = os.path.join(input_dir, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            self.dict = yaml.safe_load(f)

        self._reconstruct_interpolators()
