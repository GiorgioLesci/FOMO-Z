"""
Derive probability density functions.

This module derives probability functions based on point-estimate probabilities
obtained through the `PointProbability` class. It is ideal to derive photometric
redshifts (photo-zs) of individual galaxies, though it is general enough to estimate
distributions of other physical quantities, such as stellar masses and star formation rates.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import numpy as np
from tqdm import tqdm

from ..data.dataset import Dataset
from ..data.selection_definition import SelectionDefinition
from .point_probability import PointProbability


class ProbabilityFunction:
    """Get probability density functions."""

    def __init__(
        self,
        reference_dataset: Dataset,
        test_dataset: Dataset = None,
        random_variable="redshift",
        delta=0.05,
        min_variable=0.0,
        max_variable=3.0,
        verbose=True,
    ):
        r"""
        bla

        bla...
        The output of the `process` method is a probability density function
        (whose integral equals 1) for each object in `test_dataset`.
        A selection of the most informative features is mandatory before the `process`
        call. In the current version of the code, the following function implements
        such a selection: `select_features_grid`.

        Parameters
        ----------
        reference_dataset : Dataset
            Reference sample providing photometry, target information,
            and per-object weights. Its DataFrame must contain magnitude
            uncertainties in columns named ``<band>_error`` for every band,
            and finite, strictly positive weights in ``weights``.

            Target memberships are obtained from the reference properties
            or, when available, from the integrated redshift PDFs.
        test_dataset : Dataset, optional
            Sample for which probabilities are estimated. Its ordered
            color list must match that of `reference_dataset`, and its
            DataFrame must contain ``<band>_error`` columns.

            Test-sample weights and redshift PDFs are not used to estimate
            its probabilities. If None, the code uses `reference_dataset`
            as the test sample and applies the self-test matching convention
            (see the notes in the documentation of the constructor of
            `PointProbability`).
        random_variable : str, default="redshift"
            The random variable for which probability functions are estimated.
            The default, "redshift", implies the derivation of photo-zs.
            `random_variable` must be a key of the DataFrames contained by
            `reference_dataset` and `test_dataset`.
        delta : float, default=0.05
            The sampling precision over the random variable's domain.
            Default is 0.05.
        min_variable : float, default=0.0
            Minimum of the random variable range explored to construct the
            probability functions.
        max_variable : float, default=3.0
            Maximum of the random variable range.
        verbose : bool, default=True
            Whether to display progress bars and timing information.
            This option does not suppress warnings.
        """
        self.verbose = verbose

        if random_variable not in reference_dataset.columns:
            raise ValueError(f"{random_variable} is not a key of reference_dataset.")
        if random_variable not in test_dataset.columns:
            raise ValueError(f"{random_variable} is not a key of test_dataset.")
        
        # Build the dictionary
        self.variable_samples = np.arange(min_variable, max_variable, delta)

        self.dict_point = {}
        for s in self.variable_samples:
            # Define SelectionDefinition

            # Fill the dict
            self.dict_point[s] = PointProbability(
                selection_definition,
                reference_dataset,
                test_dataset=test_dataset,
                verbose=False,
            )

    def select_features_grid(
        self,
        tolerance=0.1,
        purity_min=0.9,
        grid_precision=0.15,
        step=0.001,
    ):
        r"""
        Select the most informative colors using a grid-based color selection.

        See the documentation of the homonymous function in `PointProbability`
        for a detailed description. This function does not reimplement that
        method. Instead, it calls `select_features_grid` for each `PointProbability`
        object created by the `ProbabilityFunction` class.

        Parameters
        ----------
        tolerance : float, default=0.1
            Maximum allowed relative variation in global purity and completeness, at
            each elimination step, for a color to be excluded from the selection.
            This is a relative tolerance, not a variation in percentage points: e.g.
            a value of 0.1 means the purity/completeness can change by at most 10%
            of its current value, not by 10 percentage points. Note that, since this
            is a relative tolerance, its effective strictness depends on how close
            purity and completeness are to zero: near-zero values make the tolerance
            very permissive in absolute terms.
        purity_min : float, default=0.9
            Required minimum value for the global purity of the initial grid-based
            selection. Default (recommended) is 90%.
        grid_precision : float, default=0.15
            Precision of the color-color grids. A smaller value yields a finer,
            more accurate grid at the cost of increased computation, RAM usage, and
            storage requirements.
        step : float, default=0.001
            Threshold reduction step applied when refining the grid selection.
        """
        return None

