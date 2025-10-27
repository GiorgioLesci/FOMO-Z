"""
OBJECT SELECTION CLASS.

This class selects objects in an input photometric sample,
applying a calibrated color selection for any redshift threshold.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import os
import re

import numpy as np
import numexpr as ne
from scipy.interpolate import BSpline
import yaml
import copy


class SelectBackground:
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

    def __init__(self):
        self.dict = {}
        self.mask = []

    def _reconstruct_interpolators(self, tol=1e-6):
        """
        Reconstruct interpolator objects.

        This method reconstructs the
        interpolator functions for each cut expression.

        After reconstruction, the `self.dict` entries will also include:
          - ``c_interp`` : `scipy.interpolate.BSpline`
              The interpolator object for the intercept parameter.
          - ``s_interp`` : `scipy.interpolate.BSpline`, optional
              The interpolator object for the slope parameter (if present).

        Parameters
        ----------
        tol : float, optional
            Small tolerance applied to the interpolation domain boundaries
            to safely allow evaluation near edges (default: 1e-6).

        Returns
        -------
        None
            Updates ``self.dict`` in place with the loaded dictionary and
            reconstructed interpolator objects.
        """
        for key, entry in self.dict.items():

            # skip metadata
            if key == "metadata":
                continue

            # reconstruct c interpolator
            if all(k in entry for k in ("knots", "order", "c_coefficients")):
                knots = np.array(entry["knots"], dtype=float)
                knots[0] -= tol
                knots[-1] += tol

                self.dict[key]["c_interp"] = BSpline(
                    knots,
                    entry["c_coefficients"],
                    entry["order"],
                    extrapolate=False,
                )

                # reconstruct s interpolator if present
                if all(k in entry for k in ("knots", "order", "s_coefficients")):
                    self.dict[key]["s_interp"] = BSpline(
                        knots,
                        entry["s_coefficients"],
                        entry["order"],
                        extrapolate=False,
                    )

    def _select_objects(self, z_l, mags):
        """
        Select objects at a given redshift threshold.

        For each cut expression in ``self.dict``, this method:
        1. Interpolates (or uses fixed) cut parameters ``c`` and ``s``
           at the given redshift.
        2. Evaluates the corresponding logical expression with the
           magnitudes in ``mags`` using `numexpr`.
        3. Combines results across all expressions with a logical OR.

        Parameters
        ----------
        z_l : float
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
        z_l_points = self.dict["metadata"]["z_l"]
        mask = np.full(len(mags[list(mags.keys())[0]]), False)
        local_dict = dict(mags)
        local_dict["c"] = None
        local_dict["s"] = None
        for expr in self.dict.keys():

            if expr == "metadata":
                continue

            if min(z_l_points) - 1.0e-6 <= z_l <= max(z_l_points) + 1.0e-6:
                # set c
                if "c_interp" in self.dict[expr]:
                    local_dict["c"] = self.dict[expr]["c_interp"](z_l)
                else:
                    local_dict["c"] = self.dict[expr]["c_optimal"][0]

                # set s
                if "s_optimal" in self.dict[expr]:
                    if "s_interp" in self.dict[expr]:
                        local_dict["s"] = self.dict[expr]["s_interp"](z_l)
                    else:
                        local_dict["s"] = self.dict[expr]["s_optimal"][0]

                # evaluate the mask
                mask |= ne.evaluate(expr, local_dict=local_dict)

        return mask

    def process(self, z_lim, dataset):
        """
        Apply optimal color selections to a dataset.

        For each redshift in ``z_lim``, this method evaluates the stored
        color conditions based on the magnitudes in ``dataset`` and produces
        a boolean mask. All results are stored in ``self.mask``.

        Parameters
        ----------
        z_lim : array-like or float
            Redshift threshold(s) at which the cuts are be applied.
        dataset : pandas.DataFrame
            Table containing the photometric dataset. Must have columns
            corresponding to the magnitude names stored in
            ``self.dict['metadata']['bands']``.

        Returns
        -------
        None
            Updates ``self.mask`` with a list of boolean arrays, one for each
            redshift in ``z_lim``.
        """
        if not self.dict:
            raise ValueError(
                "A color cut dictionary must be set " "before the call of this function"
            )

        if isinstance(z_lim, (float, int)):
            self.z_lim = np.array([float(z_lim)])
        else:
            self.z_lim = z_lim

        # Magnitude mapping
        mags = {
            band: dataset[band].to_numpy() for band in self.dict["metadata"]["bands"]
        }

        self.mask = []
        for z_l in self.z_lim:
            self.mask.append(self._select_objects(float(z_l), mags))

    def load_parameters(self, optimize, optimization_type="iterative"):
        """
        Load precomputed optimal color cut parameters
        and reconstruct interpolator objects.

        Parameters
        ----------
        optimize : Optimize
            `Optimize` object.
        optimization_type : {"single", "iterative"}, optional
            Type of optimization results. Default is "iterative".

        Returns
        -------
        None
            Updates ``self.dict`` in place with the loaded dictionary and
            reconstructed interpolator objects.
        """
        if optimization_type not in ["single", "iterative"]:
            raise ValueError(f"Optimization type '{optimization_type}' is not allowed.")

        dictionary = (
            optimize.dict_single
            if optimization_type == "single"
            else optimize.dict_iterative
        )

        self.dict = copy.deepcopy(dictionary)
        self._reconstruct_interpolators()

    def load_parameters_from_file(self, input_dir, file_name):
        """
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
