"""
Data model settings.

This module sets the target definitions for color selections,
handling any condition on galaxy properties.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

from itertools import combinations
import types

import numpy as np
import pandas as pd

from .tools import set_input_variable_precision


class SelectionDefinition:
    """Set the target selections for the derivation of color selections."""

    def __init__(
        self,
        z_lim=0.0,
        is_zlim_lower=True,
        z_lim_opposite=None,
        additional_rules=None,
    ):
        """
        Set the rules defining the target sample for color selections.

        Class defining the color selection target sample.

        Parameters
        ----------
        z_lim : float or array-like, default = 0.0
            Float or array/list of redshift limits.
            It can represent either lower of higher limit values, depending on
            the value of `is_zlim_lower`. To calibrate a selection of objects
            in a definite redshift range, use the input parameter `z_lim_opposite`.
            This defines the second edge of the redshift bin.
            If `z_gal_string=None` and no `PDZs` are provided to the `Dataset` object
            passed to the `CalibrationLinear` and `CalibrationGrid` classes, then `z_lim` is not used.
        is_zlim_lower : bool, default = True
            If True, `z_lim` is a set of lower redshift limits. Otherwise, `z_lim`
            is a set of higher limits.
        z_lim_opposite : float, default = None
            Upper (lower) redshift limit if `is_zlim_lower=True` (`is_zlim_lower=False`).
            This redshift limit is the same for all values of `z_lim`.
            If `z_lim=0.5`, `is_zlim_lower=True`, and `z_lim_opposite=1.0`,
            the targeted sample lies in the redshift range `[0.5, 1.0]`.
            The same result is obtained with
            `z_lim=1.0`, `is_zlim_lower=False`, and `z_lim_opposite=0.5`.
        additional_rules : list of lambda functions, default = None
            Additional selection rules beyond those on redshift.
            Each rule is a lambda function having a pandas DataFrame as the variable.
            All rules (including the one on the i-th `z_lim`) are
            combined through the AND logical operator.
            Example: ``[lambda df: df["logM"] > 10.0, lambda df: df["u"] < 24.0]``.
            The additional rules defined here are the same for all values of `z_lim`.

        Notes
        -----
        - If `z_lim_opposite` is defined and redshift PDFs
          were provided to the `Dataset` class, the resulting mask
          accounts for this by integrating the redshift PDFs accordingly.
        """
        self.z_lim = set_input_variable_precision(z_lim)
        for z_l in self.z_lim:
            if z_l < 0:
                raise ValueError("z_lim values cannot be negative.")

        if not isinstance(is_zlim_lower, bool):
            raise TypeError("is_zlim_lower must be a boolean (True or False).")
        self.is_zlim_lower = is_zlim_lower

        if z_lim_opposite is not None:
            if not isinstance(z_lim_opposite, (float, np.floating)):
                raise ValueError("If not None, z_lim_opposite must be a float.")
            if is_zlim_lower:
                self.z_up = z_lim_opposite
            else:
                self.z_low = z_lim_opposite
        else:
            if is_zlim_lower:
                self.z_up = 1000.0
            else:
                self.z_low = 0.0

        self.z_lim_opposite = z_lim_opposite

        for i, zz in enumerate(self.z_lim):
            if is_zlim_lower and self.z_up <= zz:
                raise ValueError(
                    f"z_lim_opposite is lower than or equal to the {i}-th z_lim value."
                )
            elif is_zlim_lower == False and self.z_low >= zz:
                raise ValueError(
                    f"z_lim_opposite is larger than or equal to the {i}-th z_lim value."
                )

        # Check the format of the additional rules
        if additional_rules is not None:

            if not isinstance(additional_rules, list):
                raise ValueError("additional_rules must be a list.")

            for i, f in enumerate(additional_rules):

                if not (isinstance(f, types.FunctionType) and f.__name__ == "<lambda>"):
                    raise ValueError(f"additional_rules[{i}] is not a lambda function.")

                if (
                    "redshift" in f.__code__.co_consts
                    or "redshift" in f.__code__.co_names
                ):
                    raise ValueError(
                        "Additional rules cannot contain operations involving 'redshift'. Use z_lim_opposite."
                    )

        self.additional_rules = additional_rules

    def _get_target_mask(self, dataset, z_lim, CDZs, PDZ_x):
        """
        Create a mask identifying galaxies that satisfy a set of conditions.

        This function generates a 2D mask array where each row corresponds
        to a different redshift limit and each column corresponds to a galaxy in the
        dataset. A value of `True` indicates that a galaxy meets all selection criteria
        for that particular redshift limit.

        Parameters
        ----------
        dataset : pandas.DataFrame
            Input dataset containing galaxy properties. Must include
            any additional properties referenced in `self.additional_rules`.
        z_lim : numpy.array of float
            Limiting redshift points.
        CDZs : array-like of array-like
            Cumulative redshift distributions (normalization IS REQUIRED) used
            in the metrics computation. If None, point-estimates are used
            (whose key is defined by the "redshift" key in the data set).
        PDZ_x : array-like
            Redshift values where CDZs[i] is computed.

        Returns
        -------
        target_mask : numpy.ndarray
            2D mask with shape (len(z_lim), len(dataset)). If no prior weights
            or redshift PDFs are provided, the mask is boolean. Otherwise, it
            contains float values accounting for redshift probabilities and/or
            prior weights.
            target_mask[i, j] indicates whether galaxy j satisfies all selection
            criteria for redshift limit z_lim[i]. For float masks, the value also
            accounts for the redshift probability and/or prior weight.

        Notes
        -----
        - The function applies all selection criteria with AND logic: a galaxy must
          satisfy the redshift condition AND all additional rules to be marked as True.
        - The returned mask can be used directly for calculating completeness, purity,
          and other selection metrics.
        - If `z_gal_string=None` and no `PDZs` are provided to the `Dataset` object
          passed to the `CalibrationLinear` and `CalibrationGrid` classes, then `z_lim` is not used.
        """
        input_weights = np.asarray(dataset["weights"].values)
        if CDZs is None and np.allclose(input_weights, 1.0):
            target_mask = np.full((len(z_lim), len(dataset)), False, dtype=bool)
        else:
            target_mask = np.full((len(z_lim), len(dataset)), 0.0, dtype=float)
            weights = input_weights

        if CDZs is not None:
            idxs = np.searchsorted(PDZ_x, z_lim, side="left")
            idxs = np.clip(idxs, 0, len(PDZ_x) - 1)

            # If an additional redshift rule is present, precompute its CDF index
            if self.z_lim_opposite is not None:
                idx_additional = np.searchsorted(
                    PDZ_x, self.z_lim_opposite, side="left"
                )
                idx_additional = np.clip(idx_additional, 0, len(PDZ_x) - 1)

        for i, zz in enumerate(z_lim):

            # Limiting redshift
            if self.is_zlim_lower and CDZs is None:
                if dataset["redshift"].isna().all():
                    target_mask[i] = np.ones_like(target_mask[i])  # select everything
                else:
                    target_mask[i] = (dataset["redshift"] > zz) & (
                        dataset["redshift"] < self.z_up
                    )

            elif self.is_zlim_lower == False and CDZs is None:
                if dataset["redshift"].isna().all():
                    target_mask[i] = np.ones_like(target_mask[i])  # select everything
                else:
                    target_mask[i] = (dataset["redshift"] < zz) & (
                        dataset["redshift"] > self.z_low
                    )

            else:

                if self.z_lim_opposite is not None:
                    if self.is_zlim_lower:
                        probs = CDZs[:, idx_additional] - CDZs[:, idxs[i]]
                    else:
                        probs = CDZs[:, idxs[i]] - CDZs[:, idx_additional]

                else:
                    if self.is_zlim_lower:
                        probs = 1 - CDZs[:, idxs[i]]
                    else:
                        probs = CDZs[:, idxs[i]]

                target_mask[i] = np.clip(probs, 0.0, 1.0)

            # Multiply by the prior weights
            if not np.issubdtype(target_mask.dtype, np.bool_):
                target_mask[i] *= weights

            # Additional rules
            if self.additional_rules is None:
                continue

            for f in self.additional_rules:
                condition = f(dataset)
                if np.issubdtype(target_mask.dtype, np.bool_):
                    target_mask[i] &= condition
                else:
                    target_mask[i] *= condition

        return target_mask
