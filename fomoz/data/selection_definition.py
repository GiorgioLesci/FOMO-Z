"""
Data model settings.

This module sets the target definitions for color selections,
handling any condition on galaxy properties.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

from itertools import combinations
import inspect

import numpy as np
import pandas as pd

from ..auxiliary import set_input_variable_precision


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
            passed to the classification classes, only the default redshift-selection
            parameters are accepted. In this case, selection depends solely on additional_rules.
            Non-default redshift-selection parameters raise a ValueError.
        is_zlim_lower : bool, default = True
            If True, `z_lim` is a set of lower redshift limits. Otherwise, `z_lim`
            is a set of higher limits.
        z_lim_opposite : float, default = None
            Upper (lower) redshift limit if `is_zlim_lower=True` (`is_zlim_lower=False`).
            This redshift limit is the same for all values of `z_lim`.
            If `z_lim=0.5`, `is_zlim_lower=True`, and `z_lim_opposite=1.0`,
            the targeted sample lies in the redshift range `(0.5, 1.0)`.
            The same result is obtained with
            `z_lim=1.0`, `is_zlim_lower=False`, and `z_lim_opposite=0.5`.
        additional_rules : list of tuples, default = None
            Additional selection rules beyond those on redshift.
            Each rule is a pair ``(columns, function)``, where ``columns``
            is a column name or a non-empty sequence of column names.

            The function receives the corresponding pandas Series as
            positional arguments, in the declared order. It must return
            one boolean value per input row, preserving row order.
            All row-dependent inputs must come from the supplied Series.

            Examples:
            ``[("logM", lambda log_mass: log_mass > 10.0)]``
            or
            ``[(("logM", "sfr"), lambda log_mass, sfr: sfr / 10**log_mass < 1e-11)]``.

            Rules are combined with each other and with the redshift
            selection through logical AND. The same additional rules
            apply to every value of `z_lim`.

            Redshift conditions must be specified through the dedicated
            redshift parameters, not through additional rules.
            Column existence and returned boolean masks are checked
            when the rules are applied to a dataset.

        Notes
        -----
        - If `z_lim_opposite` is defined and redshift PDFs
          were provided to the `Dataset` class, the resulting mask
          accounts for this by integrating the redshift PDFs accordingly.

        Warnings
        --------
        Reference properties required by the selection must not contain
        NaN, missing-value sentinels declared in `Dataset.missing_data`,
        or infinite numerical values. These entries represent unknown
        properties, not objects that fail the selection criteria.

        Invalid entries raise a `ValueError` identifying the affected columns.
        Use `Dataset.drop_missing()` to remove them before initializing
        calibration or probability objects.

        Missing photometry is handled separately and does not require
        removing an object solely because some bands are unavailable.
        Target properties need not be known for an independent test sample.
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

        # Validate and normalize additional rules
        self.additional_rules = None
        self.required_columns = ()

        if additional_rules is not None:
            if not isinstance(additional_rules, list):
                raise TypeError("additional_rules must be a list.")

            normalized_rules = []
            required_columns = []

            for i, rule in enumerate(additional_rules):
                if not isinstance(rule, tuple) or len(rule) != 2:
                    raise TypeError(
                        f"additional_rules[{i}] must be a " "(columns, function) tuple."
                    )

                columns, function = rule

                if isinstance(columns, str):
                    columns = (columns,)
                elif isinstance(columns, (list, tuple)):
                    columns = tuple(columns)
                else:
                    raise TypeError(
                        f"Columns in additional_rules[{i}] must be "
                        "a string or a sequence of strings."
                    )

                if not columns or any(
                    not isinstance(column, str) or not column.strip()
                    for column in columns
                ):
                    raise ValueError(
                        f"additional_rules[{i}] must declare at least "
                        "one non-empty column name."
                    )

                if len(set(columns)) != len(columns):
                    raise ValueError(
                        f"additional_rules[{i}] contains duplicate " "column names."
                    )

                if "redshift" in columns:
                    raise ValueError(
                        "Redshift conditions must use z_lim and "
                        "z_lim_opposite, not additional_rules."
                    )

                if not callable(function):
                    raise TypeError(
                        f"The function in additional_rules[{i}] " "must be callable."
                    )

                # Check that the declared columns can be passed as
                # positional arguments without evaluating the rule
                try:
                    signature = inspect.signature(function)
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        f"Cannot inspect the function in "
                        f"additional_rules[{i}]. "
                        "Wrap it in a lambda with explicit arguments."
                    ) from error

                try:
                    signature.bind(*([None] * len(columns)))
                except TypeError as error:
                    raise TypeError(
                        f"The function in additional_rules[{i}] "
                        f"cannot be called with {len(columns)} "
                        "positional argument(s), one per declared column."
                    ) from error

                normalized_rules.append((columns, function))

                for column in columns:
                    if column not in required_columns:
                        required_columns.append(column)

            self.additional_rules = normalized_rules
            self.required_columns = tuple(required_columns)

    def _get_target_mask(self, dataset, z_lim, CDZs, PDZ_x, z_gal_string, missing_data):
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
        z_gal_string : str or None
            Original redshift-column setting of the reference Dataset.
            The DataFrame column is already standardized to "redshift".
            If None and CDZs is None, no redshift condition is applied.
        missing_data : scalar or sequence
            Missing-value sentinels declared by the reference Dataset.
            Required selection properties are checked for these values,
            missing entries, and numerical infinities before evaluation.

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
        """
        z_lim = np.asarray(z_lim)

        if z_lim.ndim != 1 or z_lim.size == 0:
            raise ValueError("z_lim must be a non-empty one-dimensional array-like.")

        has_redshift_information = CDZs is not None or z_gal_string is not None

        has_redshift_constraint = (
            np.any(z_lim != 0.0)
            or self.z_lim_opposite is not None
            or not self.is_zlim_lower
        )

        if has_redshift_constraint and not has_redshift_information:
            raise ValueError(
                "A redshift selection was requested, but the reference Dataset "
                "provides neither point-estimate redshifts nor PDZs. "
                "Set z_gal_string to the reference redshift column, or provide "
                "PDZs and PDZ_x. For selections based only on additional_rules, "
                "leave the redshift-selection parameters at their defaults."
            )

        # Validate the reference properties required by the selection
        columns_to_check = list(self.required_columns)

        if CDZs is None and z_gal_string is not None:
            columns_to_check.append("redshift")

        unknown = [
            column for column in columns_to_check if column not in dataset.columns
        ]
        if unknown:
            raise KeyError(
                f"Selection requires columns not found in the "
                f"reference dataset: {unknown}"
            )

        if missing_data is None:
            sentinels = []
        elif np.isscalar(missing_data):
            sentinels = [missing_data]
        else:
            sentinels = list(missing_data)

        invalid_counts = {}

        for column in columns_to_check:
            values = dataset[column]
            invalid = (
                values.isna() | values.isin(sentinels) | values.isin([np.inf, -np.inf])
            )

            if invalid.any():
                invalid_counts[column] = int(invalid.sum())

        if invalid_counts:
            raise ValueError(
                "Required reference properties contain missing or "
                f"infinite values (counts per column): {invalid_counts}. "
                f"Use reference_dataset.drop_missing("
                f"{list(invalid_counts)!r}) before initializing "
                "calibration or probability objects."
            )

        # Evaluate additional rules once for all redshift thresholds
        additional_mask = np.ones(len(dataset), dtype=bool)

        for rule_index, (columns, function) in enumerate(self.additional_rules or []):
            condition = function(*(dataset[column] for column in columns))

            # Series must preserve the input index and row order
            if isinstance(condition, pd.Series):
                if not condition.index.equals(dataset.index):
                    raise ValueError(
                        f"additional_rules[{rule_index}] must preserve "
                        "the input index and row order."
                    )
            else:
                condition = np.asarray(condition)

            if condition.shape != (len(dataset),):
                raise ValueError(
                    f"additional_rules[{rule_index}] must return "
                    "one boolean value per input row."
                )

            if np.any(pd.isna(condition)):
                raise ValueError(
                    f"additional_rules[{rule_index}] returned " "missing values."
                )

            if not pd.api.types.is_bool_dtype(condition.dtype):
                raise TypeError(
                    f"additional_rules[{rule_index}] must return " "boolean values."
                )

            additional_mask &= np.asarray(condition, dtype=bool)

        # Do the rest
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
                if z_gal_string is None:
                    target_mask[i] = np.ones_like(target_mask[i])  # select everything
                else:
                    target_mask[i] = (dataset["redshift"] > zz) & (
                        dataset["redshift"] < self.z_up
                    )

            elif self.is_zlim_lower == False and CDZs is None:
                if z_gal_string is None:
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
            if np.issubdtype(target_mask.dtype, np.bool_):
                target_mask[i] &= additional_mask
            else:
                target_mask[i] *= additional_mask

        return target_mask
