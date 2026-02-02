"""
Data model settings.

This module sets the target definitions for color selections,
handling any condition on galaxy properties.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

from itertools import combinations

import numpy as np
import pandas as pd

from .tools import set_input_variable_precision


class DataModel:
    """Set the target selections for the derivation of color selections."""

    def __init__(
        self,
        z_lim,
        thresholded_quantity_min,
        maximized_quantity="completeness",
        maximized_quantity_min=0.05,
        is_zlim_lower=True,
        additional_rules=None,
        s_samples=np.linspace(0.3, 1.7, 30),
        c_precision=0.05,
        grid_precision=0.15,
        step=0.0001,
        threshold_buffer=0.05,
        alpha=0.1,
        k=1,
        do_bootstrap=True,
        n_bootstraps=1000,
    ):
        """
        Set the data model for color selections.

        Class defining the color selection data model.

        This class defines color selection criteria.

        Parameters
        ----------
        z_lim : array-like
            Array of redshift limits/thresholds to be used in selection tests.
            It can represent either lower of higher limit values, depending on
            the value of `is_zlim_lower`.
        thresholded_quantity_min : float or array-like
            Required minimum for the thresholded quantity, namely
            global purity (if `maximized_quantity == "completeness"`)
            or completeness (if `maximized_quantity == "purity"`) across all redshift bins.
            If float, the same threshold is applied at any limiting redshift point.
            If array-like, a different threshold is used at each limiting
            redshift.
        maximized_quantity : str, default="completeness"
            The quantity to maximize. Possibilities are either "completeness" or
            "purity".
        maximized_quantity_min : float, default=0.05
            Required minimum for the maximized completeness (if `maximized_quantity == "completeness"`)
            or purity (if `maximized_quantity == "purity"`) across all redshift bins.
            This is not a threshold on the global value of the maximized quantity.
            In `CalibrateLinear`, only single color conditions that meet the requirement on
            this parameter are considered. In `CalibrateGrid`, only color-color spaces that
            yield a maximized quantity larger than this parameter are considered. The default
            value is 0.05, namely 5%. This threshold is useful to exclude random fluctuations
            and reduce statistical uncertainties on selection purity and completeness.
        is_zlim_lower : bool, default = True
            If True, `z_lim` is a set of lower redshift limits. Otherwise, `z_lim`
            is a set of higher limits.
        additional_rules : list of lists, default = None
            Additional selection rules beyond redshift. Each rule is a list
            with this format: [property, operator, threshold].
            All rules (including the one on the i-th `z_lim`) are
            combined through the AND logical operator.
            Example: ``[ ["logM", ">", 10.0], ["u", "<", 24.0] ]``.
            Supported operators: ">", "<", ">=", "<=", "==", "!=".
        s_samples : array-like, default=np.linspace(0.3, 1.7, 30)
            Array of slope values for generating linear cuts in color–color space.
            These samples must be strictly positive. Used only for linear color cuts
            (i.e., used by `CalibrateLinear`, not used by `CalibrateGrid`).
        c_precision : float, default=0.05
            Precision of the c parameter sampling in `CalibrateLinear`. Smaller values
            increase accuracy but significantly increase the number of operations.
            If `c_precision` is halved, expect about 2 times more c steps per s
            (also runtime and RAM roughly double). Used only for linear color cuts
            (i.e., used by `CalibrateLinear`, not used by `CalibrateGrid`).
        grid_precision : float, default=0.15
            Precision of the grid in `CalibrateGrid`. A smaller value yields a finer,
            more accurate grid at the cost of increased computation, RAM usage, and
            storage requirements.
        step : float, default=0.0001
            Defines the threshold reduction (i) on single color conditions in `CalibrateLinear`,
            and (ii) on grid elements in `CalibrateGrid`. Default is 0.0001.
        threshold_buffer : float, default=0.05
            Used only in `CalibrateLinear`. Quantity subtracted from `thresholded_quantity_min`.
            Color conditions not satisfying the final threshold, namely
            `thresholded_quantity_min - threshold_buffer`, are not
            considered in the optimization process. This is useful to avoid filling memory
            with unused information. The default value is 0.05, namely a 5 percentage point
            subtraction. If this value is too small, the optimization method in
            `CalibrateLinear` raises an error.
        alpha : float, default=0.1
            Significance level used in `CalibrateGrid` to discard statistically
            non-robust grid bins. A bin is considered as significant only if the posterior
            probability that its true per-bin purity is below the optimized
            threshold is smaller than `alpha`. Default is 0.1, namely 10% probability.
            To avoid the bin significance classification, and include all possible bins,
            set this parameter equal to 1.
            A more detailed description can be found in the documentation of `CalibrateGrid`.
        k : int, default=1
            Used only in `CalibrateLinear`.
            Requested degree of the spline interpolation, used to interpolate color
            condition parameters. Default is 1 (linear).
            The actual degree is adjusted case by case:
            - If there are fewer than ``k+1`` points, the degree is reduced
              to ``n_points - 1``.
            - If there is only one point, interpolation is skipped.
        do_bootstrap : bool, default=True
            If True, derive uncertainties on C, P, and F through bootstrap.
        n_bootstraps : int, default=1000
            Number of bootstrap resamples to perform. It defines
            the number of resampling in error estimation.
        """
        self.z_lim = set_input_variable_precision(z_lim)

        for z_l in self.z_lim:
            if z_l < 0:
                raise ValueError("z_lim values cannot be negative")

        if isinstance(thresholded_quantity_min, float):
            self.thresholded_quantity_min = [thresholded_quantity_min] * len(self.z_lim)
        else:
            if len(thresholded_quantity_min) != len(self.z_lim):
                raise ValueError(
                    f"thresholded_quantity_min length ({len(thresholded_quantity_min)}) "
                    f"must match z_lim length ({len(self.z_lim)})"
                )
            self.thresholded_quantity_min = thresholded_quantity_min
        for t in self.thresholded_quantity_min:
            if not (0.0 <= t < 1.0):
                raise ValueError(
                    "thresholded_quantity_min accepted values lie in the range [0, 1)."
                )

        if isinstance(maximized_quantity_min, (float, int)):
            if not (0.0 <= maximized_quantity_min <= 1.0):
                raise ValueError("maximized_quantity_min must range from 0 to 1.")
            self.maximized_quantity_min = maximized_quantity_min
        else:
            raise TypeError("maximized_quantity_min must be a float or integer.")

        if np.any(s_samples <= 0):
            raise ValueError(
                "All slope values in s_samples must be strictly positive. "
                f"Found non-positive values: {s_samples[s_samples <= 0]}"
            )
        self.s_samples = s_samples.astype(np.float32)
        self.c_precision = float(c_precision)
        self.grid_precision = float(grid_precision)

        self.is_zlim_lower = is_zlim_lower
        self.additional_rules = additional_rules

        self.step = step
        if alpha < 0 or alpha > 1:
            raise ValueError(
                f"alpha must range between 0 and 1. The current value is {alpha}."
            )
        self.alpha = alpha

        if threshold_buffer > thresholded_quantity_min:
            raise ValueError(
                "threshold_buffer must be strictly smaller than thresholded_quantity_min."
            )
        self.buffered_threshold = thresholded_quantity_min - threshold_buffer

        if maximized_quantity not in ["completeness", "purity"]:
            errMsg = "maximized_quantity must be either 'completeness' or 'purity'."
            raise ValueError(errMsg)
        if maximized_quantity == "completeness":
            self.thresholded_quantity = "purity"
        else:
            self.thresholded_quantity = "completeness"

        self.k = k
        self.do_bootstrap = do_bootstrap
        self.n_bootstraps = n_bootstraps

        # Check the format of the additional rules
        if additional_rules is not None:

            if not isinstance(additional_rules, list):
                raise ValueError("additional_rules must be a list.")

            for i in range(len(additional_rules)):

                if not isinstance(additional_rules[i], list):
                    raise ValueError(f"additional_rules[{i}] is not a list.")

                if len(additional_rules[i]) != 3:
                    errMsg = "Each list in additional_rules must contain "
                    errMsg += " exactly 3 elements."
                    raise ValueError(errMsg)

                property_name, operator, _ = additional_rules[i]

                if not isinstance(property_name, str):
                    errMsg = f"Rule {i}: property_name must be a string, got {type(property_name)}"
                    raise TypeError(errMsg)
                if not isinstance(operator, str):
                    errMsg = (
                        f"Rule {i}: operator must be a string, got {type(operator)}"
                    )
                    raise TypeError(errMsg)

                valid_operators = [">", "<", ">=", "<=", "==", "!="]
                if operator not in valid_operators:
                    errMsg = f"Rule {i}: operator '{operator}' is not supported. "
                    errMsg += f"Must be one of {valid_operators}"
                    raise ValueError(errMsg)

    def get_target_mask(self, dataset, z_lim):
        """
        Create a mask identifying galaxies that satisfy a set of conditions.

        This function generates a 2D boolean mask array where each row corresponds
        to a different redshift limit and each column corresponds to a galaxy in the
        dataset. A value of `True` indicates that a galaxy meets all selection criteria
        for that particular redshift limit.

        Parameters
        ----------
        dataset : pandas.DataFrame
            Input dataset containing galaxy properties. Must include a 'redshift' column
            and any additional properties referenced in `self.additional_rules`.
        z_lim : numpy.array of float
            Limiting redshift points.

        Returns
        -------
        target_mask : numpy.ndarray of bool
            2D boolean array with shape (len(z_lim), len(dataset)).
            target_mask[i, j] = True indicates that galaxy j satisfies all selection
            criteria for redshift limit z_lim[i].

        Notes
        -----
        - The function applies all selection criteria with AND logic: a galaxy must
          satisfy the redshift condition AND all additional rules to be marked as True.
        - The returned mask can be used directly for calculating completeness, purity,
          and other selection metrics.
        """
        target_mask = np.full((len(z_lim), len(dataset)), False, dtype=bool)
        for i, zz in enumerate(z_lim):

            # Limiting redshift
            if self.is_zlim_lower:
                target_mask[i] = dataset["redshift"] > zz
            else:
                target_mask[i] = dataset["redshift"] < zz

            # Additional rules
            if self.additional_rules is None:
                continue

            for property_name, operator, threshold in self.additional_rules:

                if property_name not in dataset.columns:
                    errMsg = f"The property '{property_name}' "
                    errMsg += f"in additional_rules is not in the dataset."
                    raise KeyError(errMsg)

                data = dataset[property_name]

                if operator == ">":
                    target_mask[i] &= data > threshold
                elif operator == "<":
                    target_mask[i] &= data < threshold
                elif operator == ">=":
                    target_mask[i] &= data >= threshold
                elif operator == "<=":
                    target_mask[i] &= data <= threshold
                elif operator == "==":
                    target_mask[i] &= data == threshold
                elif operator == "!=":
                    target_mask[i] &= data != threshold

        return target_mask
