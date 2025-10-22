"""
OBJECT SELECTION CLASS.

This class selects objects in an input photometric sample,
applying a calibrated color selection and for any redshift threshold.

:Author: Giorgio Francesco Lesci <giorgio.lesci2 [at] unibo [dot] it>
"""

import os
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
        `numexpr`. The ``"bands"`` key stores the list of magnitude
        names.
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

            if key == "bands":  # skip metadata
                continue

            # reconstruct c interpolator
            if all(
                k in entry for k in ("c_knots", "c_coefficients", "c_order")
            ):
                c_knots = np.array(entry["c_knots"], dtype=float)
                c_knots[0] -= tol
                c_knots[-1] += tol

                self.dict[key]["c_interp"] = BSpline(
                    c_knots,
                    entry["c_coefficients"],
                    entry["c_order"],
                    extrapolate=False,
                )

            # reconstruct s interpolator if present
            if all(
                k in entry for k in ("s_knots", "s_coefficients", "s_order")
            ):
                s_knots = np.array(entry["s_knots"], dtype=float)
                s_knots[0] -= tol
                s_knots[-1] += tol

                self.dict[key]["s_interp"] = BSpline(
                    s_knots,
                    entry["s_coefficients"],
                    entry["s_order"],
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
            Dictionary mapping magnitude names (as in ``self.dict['bands']``)
            to numpy arrays of magnitudes for all objects.

        Returns
        -------
        numpy.ndarray of bool
            Boolean mask array of shape (n_objects,), where True indicates
            that the object satisfies at least one of the color conditions.
        """
        mask = np.full(len(mags[self.dict['bands'][0]]), False)
        local_dict = dict(mags)
        local_dict["c"] = None
        local_dict["s"] = None
        for expr in self.dict.keys():

            if expr == "bands":
                continue

            if (
                min(self.dict[expr]["z_l"]) - 1e-6
                <= float(z_l)
                <= max(self.dict[expr]["z_l"]) + 1e-6
            ):
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
            ``self.dict['bands']``.

        Returns
        -------
        None
            Updates ``self.mask`` with a list of boolean arrays, one for each
            redshift in ``z_lim``.
        """
        if not self.dict:
            raise ValueError(
                "A color cut dictionary must be set "
                "before the call of this function"
            )

        if (
            "c" in self.dict["bands"]
            or "s" in self.dict["bands"]
        ):
            raise ValueError("Magnitude keys cannot equal 'c' or 's'.")

        if isinstance(z_lim, (float, int)):
            self.z_lim = np.array([float(z_lim)])
        else:
            self.z_lim = z_lim
        
        self.mask = []
        mags = {band: dataset[band].to_numpy() for band in self.dict['bands']}
        for z_l in self.z_lim:
            self.mask.append(self._select_objects(z_l, mags))

    def load_parameters(self, input_dict):
        """
        Load precomputed optimal color cut parameters
        and reconstruct interpolator objects.

        Parameters
        ----------
        input_dict : dict
            Dictionary containing optimal cut parameters.

        Returns
        -------
        None
            Updates ``self.dict`` in place with the loaded dictionary and
            reconstructed interpolator objects.
        """
        self.dict = copy.deepcopy(input_dict)
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
