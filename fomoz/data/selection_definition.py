"""
Define target-sample criteria for classification and reconstruction.

Combine an optional redshift selection with rules on numerical or categorical
properties, and evaluate the corresponding prior-weighted target memberships.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

from itertools import combinations
import inspect

import numpy as np
import pandas as pd

from ..auxiliary import set_input_variable_precision


class SelectionDefinition:
    """
    Define redshift constraints and additional target-selection rules.

    Additional rules are combined through logical AND and applied at every
    redshift threshold. For reference samples, redshift membership is evaluated
    from point estimates or cumulative PDZs and multiplied by the object prior
    weights. This class specifies the target criteria; it does not fit color
    cuts or estimate probabilities from photometry.

    Attributes
    ----------
    z_lim : numpy.ndarray
        Redshift thresholds converted to float32 and rounded to six decimal
        places by ``set_input_variable_precision``.
    is_zlim_lower : bool
        Whether the thresholds are lower rather than upper bounds.
    z_lim_opposite : float or None
        Explicit opposite boundary shared by all thresholds, if supplied.
    z_up : float
        Upper boundary for point estimates when ``is_zlim_lower=True``.
        Defaults to 1000.0. Only defined in this mode.
    z_low : float
        Lower boundary for point estimates when ``is_zlim_lower=False``.
        Defaults to 0.0. Only defined in this mode.
    additional_rules : list or None
        Validated ``(columns, function)`` pairs, with each ``columns`` entry
        normalized to a tuple. None if no rules were supplied.
    required_columns : tuple of str
        Unique columns declared by additional rules, in first-appearance
        order. Does not include the automatically required point-redshift
        column.
    """

    def __init__(
        self,
        z_lim=0.0,
        is_zlim_lower=True,
        z_lim_opposite=None,
        additional_rules=None,
    ):
        """
        Initialize the criteria defining the target sample.

        Parameters
        ----------
        z_lim : float or one-dimensional array-like, default=0.0
            Non-negative redshift thresholds. Each threshold defines a separate
            selection using the same additional rules and opposite boundary.
            Values are converted to float32 and rounded to six decimal places.
            Supply a non-empty array when specifying multiple thresholds.
        is_zlim_lower : bool, default=True
            If True, each threshold is a lower redshift bound; otherwise it
            is an upper bound. For point estimates, comparisons are strict:
            objects exactly on either boundary are excluded.
        z_lim_opposite : float or None, default=None
            Common upper bound when ``is_zlim_lower=True``, or common lower
            bound otherwise. It must be strictly above or below, respectively,
            every processed threshold. Supply a float, such as ``1.0``;
            integer values are not accepted for this parameter.

            If omitted, point-estimate selections use an upper bound of
            1000.0 or a lower bound of 0.0, respectively. PDZ selections
            instead use the full probability above or below the threshold,
            without applying these implicit point-estimate bounds.
        additional_rules : list of tuples or None, default=None
            Additional criteria, each specified as ``(columns, function)``.
            ``columns`` must be a non-empty column name or a non-empty list
            or tuple of distinct column names. Numerical and categorical
            columns are supported.

            The callable receives the corresponding pandas Series as
            positional arguments, in the declared order. It must return a
            one-dimensional boolean mask with one value per input row and
            no missing entries. A returned Series must preserve the input
            index and row order; array-like results are interpreted by
            position. Scalars and numerical masks containing 0 and 1 are
            not accepted. All row-dependent inputs should come from the
            supplied Series, so the rule remains valid after row filtering.

            All rules are combined with each other and the redshift criterion
            through logical AND. None or an empty list imposes no additional
            restriction. The same rules apply to every redshift threshold.

            Use the dedicated redshift parameters for redshift constraints.
            The standardized column name ``"redshift"`` is explicitly
            forbidden in additional rules. Callable signatures are checked
            at initialization; column contents and output masks are checked
            when the rules are evaluated.

        Raises
        ------
        TypeError
            If ``is_zlim_lower`` is not a Python bool, the rule containers
            have invalid types, or a rule function is not callable, cannot
            be inspected, or cannot accept the declared positional inputs.
        ValueError
            If a threshold is negative, the opposite bound has an invalid
            type or ordering, or rule column declarations are empty,
            duplicated, invalid, or include ``"redshift"``.

        Notes
        -----
        The selection weight is the prior object weight multiplied by the
        redshift membership and by the boolean conjunction of additional
        rules. PDZ membership uses differences of the normalized cumulative
        redshift distributions stored by ``Dataset``. PDZs take precedence
        over point redshifts when both are supplied.

        A selection based only on additional properties requires a Dataset
        with ``z_gal_string=None`` and no PDZs, together with the default
        redshift parameters. Non-default redshift parameters without redshift
        information raise an error when the selection is evaluated. Default
        parameters do not disable redshift selection when redshift information
        is available: point estimates must satisfy ``0 < z < 1000``, while
        PDZ membership is the probability above zero.

        Required properties are validated across all input rows before any
        rule is evaluated, including rows with zero prior weight or rows that
        would fail another rule. Missing entries, declared missing-value
        sentinels, and numerical infinities raise ValueError; they are not
        interpreted as failed membership. Use ``Dataset.drop_missing`` on the
        required columns before initializing calibration or probability
        objects. Unused photometric bands may still contain missing values.

        Target properties need not be known for an independent test sample
        when predicting membership. Population reconstruction instead uses an
        observed selection applied to both samples; those observed properties
        must therefore be available wherever that selection is evaluated.

        Examples
        --------
        Define an open point-redshift interval, ``0.5 < z < 1.0``:

        >>> selection = SelectionDefinition(z_lim=0.5, z_lim_opposite=1.0)

        Combine multiple lower thresholds with a stellar-mass requirement:

        >>> selection = SelectionDefinition(
        ...     z_lim=[0.2, 0.4, 0.6],
        ...     additional_rules=[("logM", lambda mass: mass > 10.0)],
        ... )

        Combine numerical and categorical rules. To omit the redshift
        constraint, use this definition with a Dataset that has neither
        designated point redshifts nor PDZs:

        >>> selection = SelectionDefinition(additional_rules=[
        ...     (("logM", "sfr"), lambda mass, sfr: sfr / 10**mass < 1e-11),
        ...     ("object_type", lambda kind: kind == "galaxy"),
        ... ])
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
        Evaluate prior-weighted target membership for each redshift threshold.

        Parameters
        ----------
        dataset : pandas.DataFrame
            Internal table from a Dataset, including ``"weights"`` and all
            columns declared by additional rules. Point-redshift selections
            use the standardized ``"redshift"`` column.
        z_lim : one-dimensional array-like
            Non-empty sequence of thresholds to evaluate. These normally
            come from ``self.z_lim``. This method does not round thresholds
            or repeat the constructor's boundary-ordering checks.
        CDZs : array-like of shape (n_objects, n_samples) or None
            Normalized cumulative redshift distributions aligned with the
            table rows. Use the ``CDZs`` produced by Dataset. When supplied,
            these take precedence over point-redshift values.
        PDZ_x : array-like of shape (n_samples,) or None
            Common redshift grid for ``CDZs``. Required in PDZ mode and
            ignored otherwise. Use the grid validated by Dataset.
        z_gal_string : str or None
            Original Dataset setting identifying a point-redshift column.
            A non-None value enables point-redshift selection when CDZs
            are absent; the values are read from ``"redshift"``, not from
            this original column name. None together with absent CDZs
            disables redshift selection only for default redshift settings.
        missing_data : scalar, sequence, or None
            Missing-value sentinels declared by Dataset. None means no
            additional sentinels. Missing entries and numerical infinities
            are always rejected in required selection properties.

        Returns
        -------
        target_mask : numpy.ndarray of shape (n_thresholds, n_objects)
            Selection weights, with one row per threshold and one column
            per object. Each entry is the prior weight times the redshift
            membership times the conjunction of additional rules.

            The result is boolean when CDZs are absent and all input weights
            are close to one according to ``numpy.allclose``. In that branch,
            weights are treated as unity. Otherwise the result is floating
            point, even if individual entries happen to be zero or one.
            Weighted entries may exceed one; this is not an array of
            standalone membership probabilities.

        Raises
        ------
        KeyError
            If a required property column or ``"weights"`` is missing.
        ValueError
            If thresholds are not a non-empty one-dimensional array,
            a redshift constraint has no supporting redshift information,
            required properties contain missing or infinite values, or
            a rule returns an invalid shape, index, or missing entries.
        TypeError
            If a rule returns a mask with a non-boolean dtype.

        Notes
        -----
        Additional rules are evaluated once and reused across thresholds.
        With no redshift information and default redshift settings, redshift
        membership is one. With no additional rules, their contribution is
        one. Point-redshift membership uses strict inequalities at both ends.

        In PDZ mode, each cumulative distribution is linearly interpolated
        at the requested boundaries, with cumulative values of zero and one
        below and above the grid. Explicit intervals use CDF differences;
        a single lower or upper bound uses the upper or lower tail,
        respectively. Probabilities are clipped to [0, 1] before multiplication
        by prior weights. This follows the cumulative-PDZ convention used
        by the distribution helpers.

        Input weights and cumulative distributions are assumed to have been
        validated by Dataset. The returned values already include prior
        weights: do not multiply them by those weights a second time.
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
            # Evaluate reference CDFs at all requested redshift thresholds
            cdf_limits = np.array(
                [np.interp(z_lim, PDZ_x, cdf, left=0.0, right=1.0) for cdf in CDZs]
            ).reshape(len(dataset), len(z_lim))

            # The opposite boundary is shared by all thresholds
            if self.z_lim_opposite is not None:
                cdf_opposite = np.array(
                    [
                        np.interp(
                            self.z_lim_opposite,
                            PDZ_x,
                            cdf,
                            left=0.0,
                            right=1.0,
                        )
                        for cdf in CDZs
                    ]
                )

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
                        probs = cdf_opposite - cdf_limits[:, i]
                    else:
                        probs = cdf_limits[:, i] - cdf_opposite
                else:
                    if self.is_zlim_lower:
                        probs = 1.0 - cdf_limits[:, i]
                    else:
                        probs = cdf_limits[:, i]

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

