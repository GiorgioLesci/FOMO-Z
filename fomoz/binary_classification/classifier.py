"""
Apply calibrated linear cuts and color--color grid masks to photometry.

Load calibration objects or exported files and select objects at requested
redshift thresholds while preserving reference-grid boundaries.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import copy
import os
import re
import itertools

import numexpr as ne
import numpy as np
import yaml
from astropy.io import fits
from scipy.interpolate import BSpline

from ..auxiliary import (
    set_input_variable_precision,
    get_color_color_mask,
    update_dict_metadata,
    update_dict,
)
from .linear_classification import LinearClassification
from .grid_classification import GridClassification


class Classifier:
    """
    Apply precomputed photometric selections to a test sample.
    """

    def __init__(self):
        """
        Initialize an empty classifier.

        Attributes
        ----------
        dict_linear : dict
            Loaded linear-cut expressions, metadata, and reconstructed
            parameter interpolators. Empty until a linear calibration is loaded.
        dict_grid : dict
            Loaded grid masks, metadata including reference color limits,
            and object-bin mappings updated on each process call.
        mask : list of numpy.ndarray
            Boolean masks, one per requested threshold, after process.
        bands : list of str
            Magnitude columns required by the applied calibrations.
        z_lim : numpy.ndarray
            Requested thresholds, set by process as float32 rounded to
            six decimal places.
        """
        self.dict_linear = {}
        self.dict_grid = {}
        self.mask = []
        self.bands = []

    def _reconstruct_interpolators(self):
        """
        Reconstruct BSpline objects for the loaded linear cuts.

        Returns
        -------
        None
            Add c_interp and, for inclined cuts, s_interp to dict_linear
            entries containing a spline order. Single-point entries retain
            scalar coefficients. Extrapolation is disabled in BSpline.

        Raises
        ------
        ValueError
            If bands equal reserved parameter names or required knot or
            coefficient fields are missing. Spline-construction errors
            also propagate.
        """
        bands = self.dict_linear["metadata"]["bands"]
        if "c" in bands or "s" in bands:
            raise ValueError("Band keys cannot equal 'c' or 's'.")

        for key, entry in self.dict_linear.items():

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

    def _linear_selection(self, z_lim, mags, color_condition="all"):
        """
        Evaluate requested linear-cut expressions at one threshold.

        Parameters
        ----------
        z_lim : float
            Requested threshold. Boundary behavior is applied separately
            to each expression's calibrated range.
        mags : dict
            Band names mapped to aligned arrays of object magnitudes.
        color_condition : str, default="all"
            Combine all expressions through OR, or evaluate one loaded key.

        Returns
        -------
        numpy.ndarray of bool
            Objects satisfying at least one evaluated expression.

        Notes
        -----
        For lower-limit calibrations, requests below a cut's calibrated
        range use its minimum threshold; requests above its range receive
        no contribution from that cut. For upper-limit calibrations, requests
        above the range use its maximum threshold; requests below it receive
        no contribution. Thus, an unsupported cut contributes False, not
        an unrestricted selection. Parameters within the supported range
        are interpolated, or taken from fixed coefficients for a single
        calibration point. Comparisons with NaN colors do not pass.
        """
        all_conditions = [k for k in self.dict_linear if k != "metadata"]
        if color_condition == "all":
            color_condition = all_conditions
        else:
            if color_condition not in all_conditions:
                raise ValueError(
                    f"Not allowed color_condition value {color_condition}."
                )
            color_condition = [color_condition]

        mask = np.full(len(mags[list(mags.keys())[0]]), False)
        local_dict = dict(mags)
        local_dict["c"] = None
        local_dict["s"] = None
        for expr, expr_data in self.dict_linear.items():

            # skip not requested color conditions
            if expr not in color_condition:
                continue

            # adjust z_l
            z_l_points = expr_data["knots"]

            z_l = z_lim
            if self.dict_linear["metadata"]["z_l_is_lower_limit"] and z_lim < min(
                z_l_points
            ):
                z_l = min(z_l_points)
            elif not self.dict_linear["metadata"]["z_l_is_lower_limit"] and z_lim > max(
                z_l_points
            ):
                z_l = max(z_l_points)

            # apply the selection
            condition = (
                z_l <= max(z_l_points)
                if self.dict_linear["metadata"]["z_l_is_lower_limit"]
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

    def process(
        self,
        z_lim,
        dataset,
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
    ):
        """
        Apply loaded photometric selections to a DataFrame.

        Parameters
        ----------
        z_lim : float or one-dimensional array-like
            Requested thresholds, converted to float32 and rounded to six
            decimal places. Grid calibration thresholds must be ordered
            increasingly in the loaded metadata.
        dataset : pandas.DataFrame
            Test photometry with a column for every required magnitude band.
            Missing bands may be supplied as columns filled with NaN or a
            declared sentinel. The input DataFrame is copied, not modified.
        missing_data : scalar or list, optional
            Missing-value sentinels replaced with NaN in required bands.
            Defaults to [99, -99, 999, -999, 99.9, -99.9, 999.9, -999.9].

        Returns
        -------
        None
            Replace mask with one boolean array per requested threshold,
            preserving the input row order. If both calibration types are
            loaded, their selections are combined through logical OR.

        Raises
        ------
        ValueError
            If no calibration is loaded, required bands are absent, loaded
            linear and grid band lists differ, or reference grid limits
            are missing.

        Notes
        -----
        Linear parameters are interpolated within each cut's calibrated
        range. For grid selections, a request between adjacent thresholds
        uses the higher threshold for lower-limit calibrations and the lower
        threshold for upper-limit calibrations. Grid comparisons use a
        redshift tolerance of 1e-6.

        Below the calibrated range, lower-limit selections use the lowest
        available threshold; above the range they select no objects. For
        upper-limit selections, requests above the range use the highest
        available threshold; below the range they select no objects. Linear
        boundary handling is applied separately to each expression.

        Grid limits always remain those of the reference calibration. Only
        object-bin assignments are recomputed for the test sample. Objects
        outside a grid, on its final upper edge, or missing either color
        are excluded from that space; another space can still select them.
        Neither target properties nor magnitude errors are used here.
        """
        # First checks
        self.z_lim = set_input_variable_precision(z_lim)

        if not self.dict_linear and not self.dict_grid:
            raise ValueError(
                "A color cut dictionary must be set before the call of this function."
            )

        # Determine required bands
        if self.dict_linear and self.dict_grid:
            required_bands = self.dict_linear["metadata"]["bands"]
            if required_bands != self.dict_grid["metadata"]["bands"]:
                raise ValueError(
                    "The two input selections are calibrated on different sets of photometric bands."
                )
        elif self.dict_linear:
            required_bands = self.dict_linear["metadata"]["bands"]
        elif self.dict_grid:
            required_bands = self.dict_grid["metadata"]["bands"]

        missing_bands = [band for band in required_bands if band not in dataset.columns]
        if missing_bands:
            raise ValueError(
                f"Dataset is missing required bands: {missing_bands}. "
                f"Available columns: {list(dataset.columns)}. If you want "
                "to use this selection without a given band, create columns "
                "in the source dataset for each missing band. Fill such columns "
                "with magnitude missing values (see the documentation of this method)."
            )

        self.bands = required_bands

        # Handle missing data
        dataset = dataset.copy()
        dataset[required_bands] = dataset[required_bands].replace(missing_data, np.nan)
        N_objects = len(dataset)

        # Magnitude mapping
        mags = {band: dataset[band].to_numpy() for band in required_bands}

        # In case of grid selections, update the dictionary
        if self.dict_grid:
            if "color_limits" not in self.dict_grid["metadata"]:
                raise ValueError(
                    "Grid calibration lacks reference color limits. "
                    "Reload a calibration exported with grid limits; "
                    "the limits cannot be inferred from the test sample."
                )
            self.dict_grid = update_dict(
                self.dict_grid,
                dataset,
                self.dict_grid["metadata"],
                reset_dictionary=False,
            )

        # Apply the selection
        self.mask = []
        for z_l in self.z_lim:

            mask_linear = np.full(N_objects, False)
            mask_grid = np.full(N_objects, False)

            if self.dict_linear:
                mask_linear = self._linear_selection(z_l, mags)

            if self.dict_grid:

                tol = 1e-6
                z_l_calib = np.asarray(self.dict_grid["metadata"]["z_l"])
                z_min = np.min(z_l_calib) - tol
                z_max = np.max(z_l_calib) + tol
                i_z = None

                if self.dict_grid["metadata"]["z_l_is_lower_limit"]:
                    if z_l < z_min:
                        i_z = 0
                    elif z_l > z_max:
                        i_z = None  # No selection
                    else:
                        indices = np.where(z_l_calib >= z_l - tol)[0]
                        i_z = indices[0] if len(indices) > 0 else None
                else:
                    if z_l < z_min:
                        i_z = None  # No selection
                    elif z_l > z_max:
                        i_z = len(z_l_calib) - 1
                    else:
                        indices = np.where(z_l_calib <= z_l + tol)[0]
                        i_z = indices[-1] if len(indices) > 0 else None

                if i_z is not None:
                    mask_grid = get_color_color_mask(
                        i_z,
                        [k for k in self.dict_grid if k != "metadata"],
                        self.dict_grid,
                    )

            self.mask.append(mask_linear | mask_grid)

    def load_parameters(self, calibration):
        """
        Load a calibration object into the classifier.

        Parameters
        ----------
        calibration : LinearClassification or GridClassification
            Successfully processed calibration. Its dictionary is deep-copied,
            so subsequent classifier operations do not modify the source.

        Returns
        -------
        None
            Replace the dictionary for the supplied calibration type.
            A previously loaded calibration of the other type is retained.
            Linear interpolators are reconstructed; grid reference limits
            are preserved as part of the copied metadata.

        Raises
        ------
        ValueError
            If calibration is not a supported class. Errors reconstructing
            incomplete linear parameters also propagate.
        """
        dictionary = calibration.dict

        if isinstance(calibration, LinearClassification):
            self.dict_linear = copy.deepcopy(dictionary)
            self._reconstruct_interpolators()
            print("LinearClassification parameters successfully loaded.")
        elif isinstance(calibration, GridClassification):
            self.dict_grid = copy.deepcopy(dictionary)
            print("GridClassification parameters successfully loaded.")
        else:
            raise ValueError("Wrong type of the calibration parameter.")

    def load_parameters_from_file(self, input_dir, file_name):
        """
        Load an exported linear or grid calibration.

        Parameters
        ----------
        input_dir : str or path-like
            Directory containing the calibration file.
        file_name : str
            Filename including extension. Supported extensions are .yaml
            for linear cuts and .fits for grid masks, case-insensitively.

        Returns
        -------
        None
            Replace the corresponding calibration dictionary, retaining any
            loaded calibration of the other type. Linear interpolators are
            reconstructed. Grid masks and reference limits are restored;
            object-bin assignments are built by process on the test sample.

        Raises
        ------
        ValueError
            If the file extension is unsupported, a grid file lacks reference
            color limits or masks, saved limits are invalid or inconsistent,
            or mask shapes disagree with thresholds or grid dimensions.
            File-reading and missing-metadata errors also propagate.

        Notes
        -----
        Grid FITS files require ZL_VALUES, BANDS, ZLISLOW, GRIDPREC, and
        MASK_ image extensions with band identifiers and C1MIN, C1MAX,
        C2MIN, C2MAX headers. The saved limits and GRIDPREC reconstruct
        the original grids; limits are never estimated from test photometry.
        Legacy FITS files without these limits are rejected. Re-export the
        original calibration object using the updated writer or recalibrate.
        A failed grid load leaves the previous grid dictionary unchanged.
        """
        file_path = os.path.join(input_dir, file_name)
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        if ext == ".yaml":
            with open(file_path, "r", encoding="utf-8") as f:
                self.dict_linear = yaml.safe_load(f)

            self._reconstruct_interpolators()

            print("LinearClassification parameters successfully loaded from file.")

        elif ext == ".fits":
            with fits.open(file_path) as hdul:
                primary_header = hdul[0].header
                z_l = list(hdul["ZL_VALUES"].data["z_l"])
                z_l_is_lower_limit = bool(primary_header["ZLISLOW"])
                bands = primary_header["BANDS"].split()
                grid_precision = primary_header["GRIDPREC"]

                color_spaces = []
                color_limits = {}
                dictionary = {}

                for hdu in hdul:
                    if not hdu.name.startswith("MASK_"):
                        continue

                    cs_str = hdu.name[5:]  # strip "MASK_" prefix
                    mask_hdu = hdu

                    cs_key = (
                        (
                            mask_hdu.header["C1BAND1"].strip(),
                            mask_hdu.header["C1BAND2"].strip(),
                        ),
                        (
                            mask_hdu.header["C2BAND1"].strip(),
                            mask_hdu.header["C2BAND2"].strip(),
                        ),
                    )

                    for axis, color in enumerate(cs_key, start=1):
                        names = (f"C{axis}MIN", f"C{axis}MAX")
                        if any(name not in mask_hdu.header for name in names):
                            raise ValueError(
                                "This grid FITS file lacks reference color limits. "
                                "Re-export the original GridClassification object "
                                "with the updated writer, or recalibrate."
                            )
                        bounds = tuple(float(mask_hdu.header[name]) for name in names)
                        if not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
                            raise ValueError(f"Invalid grid limits for color {color}.")
                        if color in color_limits and color_limits[color] != bounds:
                            raise ValueError(
                                f"Inconsistent grid limits for color {color}."
                            )
                        color_limits[color] = bounds

                    mask_3d = np.asarray(mask_hdu.data, dtype=bool)
                    if mask_3d.ndim != 3 or mask_3d.shape[0] != len(z_l):
                        raise ValueError(
                            "Grid mask shape does not match calibration thresholds."
                        )
                    expected_shape = tuple(
                        int(
                            (color_limits[color][1] - color_limits[color][0])
                            / grid_precision
                        )
                        for color in cs_key
                    )
                    if mask_3d.shape[1:] != expected_shape:
                        raise ValueError(
                            "Grid mask shape does not match its saved color limits."
                        )
                    mask_dict = {i: mask_3d[i] for i in range(mask_3d.shape[0])}

                    dictionary[cs_key] = {
                        "mask": mask_dict,
                    }

                    color_spaces.append(cs_key)

                colors = list(
                    {cs[0] for cs in color_spaces} | {cs[1] for cs in color_spaces}
                )

                if not color_spaces:
                    raise ValueError("The FITS file contains no grid masks.")
                dictionary = update_dict_metadata(
                    dictionary,
                    bands,
                    colors,
                    color_spaces,
                    grid_precision,
                    z_l,
                    z_l_is_lower_limit,
                )
                dictionary["metadata"]["color_limits"] = color_limits
                self.dict_grid = dictionary

            print("GridClassification parameters successfully loaded from file.")

        else:
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                "Only '.yaml' and '.fits' files are supported."
            )
