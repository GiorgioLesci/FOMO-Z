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


class TargetDefinition:
    """Set the target selections for the derivation of color selections."""

    def __init__(
        self,
        z_lim,
        is_zlim_lower=True,
        additional_rules=None,
    ):
        """
        Set the rules defining the target sample for color selections.

        Class defining the color selection target sample.

        Parameters
        ----------
        z_lim : float or array-like
            Float or array of redshift limits.
            It can represent either lower of higher limit values, depending on
            the value of `is_zlim_lower`.
        is_zlim_lower : bool, default = True
            If True, `z_lim` is a set of lower redshift limits. Otherwise, `z_lim`
            is a set of higher limits.
        additional_rules : list of lists, default = None
            Additional selection rules beyond `z_lim`. Each rule is a list
            with this format: [property, operator, threshold].
            All rules (including the one on the i-th `z_lim`) are
            combined through the AND logical operator.
            Example: ``[ ["logM", ">", 10.0], ["u", "<", 24.0] ]``.
            Supported operators: ">", "<", ">=", "<=", "==", "!=".
        """
        self.z_lim = set_input_variable_precision(z_lim)
        for z_l in self.z_lim:
            if z_l < 0:
                raise ValueError("z_lim values cannot be negative.")

        if not isinstance(is_zlim_lower, bool):
            raise TypeError("is_zlim_lower must be a boolean (True or False).")
        self.is_zlim_lower = is_zlim_lower

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
            Input dataset containing galaxy properties. Must include a 'redshift' column
            and any additional properties referenced in `self.additional_rules`.
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
            2D mask with shape (len(z_lim), len(dataset)).
            target_mask[i, j] = True indicates that galaxy j satisfies all selection
            criteria for redshift limit z_lim[i].

        Notes
        -----
        - The function applies all selection criteria with AND logic: a galaxy must
          satisfy the redshift condition AND all additional rules to be marked as True.
        - The returned mask can be used directly for calculating completeness, purity,
          and other selection metrics.
        """
        if CDZs is None:
            dtype = bool
        else:
            dtype = float
            idxs = np.searchsorted(PDZ_x, z_lim, side="left")
            idxs = np.clip(idxs, 0, len(PDZ_x)-1)

        target_mask = np.full((len(z_lim), len(dataset)), False, dtype=dtype)

        for i, zz in enumerate(z_lim):

            # Limiting redshift
            if self.is_zlim_lower and CDZs is None:
                target_mask[i] = dataset["redshift"] > zz
            
            elif self.is_zlim_lower == False and CDZs is None:
                target_mask[i] = dataset["redshift"] < zz

            else:

                if self.is_zlim_lower:
                    probs = 1 - CDZs[:, idxs[i]]
                else:
                    probs = CDZs[:, idxs[i]]

                target_mask[i] = probs

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
                    target_mask[i] *= data > threshold
                elif operator == "<":
                    target_mask[i] *= data < threshold
                elif operator == ">=":
                    target_mask[i] *= data >= threshold
                elif operator == "<=":
                    target_mask[i] *= data <= threshold
                elif operator == "==":
                    target_mask[i] *= data == threshold
                elif operator == "!=":
                    target_mask[i] *= data != threshold

        return target_mask
