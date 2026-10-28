r"""
Reconstruct selected test-sample distributions by transferring reference weights.

The observed interval defines sample membership and feature selection. Photometric
matching then transfers each supported test object's weight to compatible selected
reference objects. Point measurements, or reference PDZ bin integrals for redshift,
provide the reconstructed distribution. No individual membership-probability
factor is applied to the transferred weights.
"""

from copy import copy
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import warnings

import numpy as np
from tqdm import tqdm

from ..data.dataset import Dataset
from ..data.selection_definition import SelectionDefinition
from .point_probability import PointProbability


class Reconstruction:
    r"""
    Reconstruct population distributions using color-compatible reference objects.

    Parameters
    ----------
    reference_dataset, test_dataset : Dataset
        Full reference and test samples, before applying the observed interval.
        Both must contain positive-weight objects inside and outside that interval.
        The same Dataset instance or underlying DataFrame cannot be used for both.
        Provide compatible photometric bands, with magnitudes and uncertainties
        in ``<band>`` and ``<band>_error`` columns for the retained colors.

        Reference ``weights`` are the prior weights :math:`w_{\rm prior}^j` of
        possible analogues. Test ``weights`` are the population contributions
        :math:`w_{\rm test}^i`. Dataset supplies unit weights by default.
    selection_variable : str
        Numerical observed property :math:`s`, available in both samples with the
        same physical meaning. Membership always uses its point values, even if
        Dataset contains PDZs. Missing or nonfinite selection values must be
        removed before constructing this object.

        This should differ from the property :math:`x` requested later through
        `get_distribution`; that distinction is a scientific recommendation,
        not an enforced name check. For example, select a photometric-redshift
        interval and reconstruct spectroscopic redshift from the reference sample.
    selection_min, selection_max : float
        Finite bounds, with selection_min < selection_max. The observed rule is
        ``selection_min < s <= selection_max`` in both samples and during feature
        calibration. It does not restrict the reconstructed property or its bins.
    verbose : bool, default=True
        Print selection counts, matching progress and distribution information.
        Warnings and validation errors remain active when False.

    Attributes
    ----------
    colors : list of tuple(str, str) or None
        Retained colors, defining :math:`X_{\rm sel}`; None before feature selection.
    discarded_colors : list of tuple(str, str)
        Colors removed by the last successful feature-selection call.
    reference_indices, test_indices : numpy.ndarray
        Selected row positions in the original Dataset DataFrames, not index labels.
    reference_weights : numpy.ndarray or None
        Accumulated weights :math:`w_{\rm rec}^j`, available after successful
        `process`. Aligned with the full reference DataFrame; entries outside
        the selected subset are zero. Their sum equals supported test weight.
    diagnostics : dict or None
        Support, match counts, effective match counts and test-weight coverage;
        see `process` for all entries and their indexing.

    Notes
    -----
    The observed sample-selection rule is

    .. math::

        \mathcal R_{\rm obs}: s_{\min}<s\leq s_{\max}.

    Call `select_features_grid` on the full reference sample to identify colors
    that distinguish objects inside this interval from those outside it. Then
    call `process`, which matches only the selected test and reference subsets,
    :math:`\Psi_{\mathcal R_{\rm obs}}` and
    :math:`\Phi_{\mathcal R_{\rm obs}}`. Finally call `get_distribution` for any
    supported numerical reference property and bin edges. The same matches and
    transferred weights are reused across properties and bin choices.

    For test object :math:`i`, let :math:`\chi_i\subseteq X_{\rm sel}` contain
    retained colors with finite test measurements and uncertainties. Its
    compatible reference set is

    .. math::

        \Sigma_i=\left\{j\in\Phi_{\mathcal R_{\rm obs}}:
        |c_{i,k}-c_{j,k}|\leq\epsilon_{ij,k}\quad\forall k\in\chi_i\right\},
        \qquad
        \epsilon_{ij,k}=\sqrt{\sigma_{i,k}^2+\sigma_{j,k}^2}.

    Reference measurements and errors must be usable in every compared color.
    For :math:`c_k=m_a-m_b`, the propagated error is
    :math:`\sigma_k=\sqrt{\sigma_{m_a}^2+\sigma_{m_b}^2}`. The propagation assumes
    uncorrelated band errors and independent reference/test measurements.
    Covariances between colors sharing a band are not modeled. Distinct Dataset
    instances do not guarantee statistically independent catalogues: overlapping
    objects or copied catalogues are not detected.

    A positive-weight selected test object is supported if it has at least
    ``min_colors_used`` usable colors and at least one compatible reference
    object with positive prior weight. It distributes its entire weight among
    those analogues:

    .. math::

        w_{\rm rec}^j=
        \sum_{\substack{i\in\Psi_{\mathcal R_{\rm obs}}^{\rm supported}\\
                        j\in\Sigma_i}}
        w_{\rm test}^i\,
        \frac{w_{\rm prior}^j}{\sum_{k\in\Sigma_i}w_{\rm prior}^k}.

    Zero-weight references contribute nothing. Consequently,
    :math:`\sum_j w_{\rm rec}^j=\sum_{i\in\Psi_{\mathcal R_{\rm obs}}^{\rm supported}}
    w_{\rm test}^i`, independently of the number of analogues per test object.
    With unit weights, each match receives :math:`1/|\Sigma_i|` from that test
    object. The accumulated weights already include the reference prior and test
    weights; do not multiply them by either again. No estimated membership
    probability is used as an extra factor.

    For point estimates, `get_distribution` computes

    .. math::

        N_b=\sum_{j\in\Phi_{\mathcal R_{\rm obs}}}
            w_{\rm rec}^j\,\delta^j(x\in B_b),
        \qquad
        n_b=\frac{N_b}{(e_{b+1}-e_b)\sum_c N_c}.

    Here :math:`B_b=[e_b,e_{b+1})`, except that the last bin also includes its
    right edge. The indicator :math:`\delta^j` is one when the reference point
    value lies in the bin and zero otherwise. ``normalize=False`` returns
    :math:`N_b`; the default returns the density :math:`n_b`, normalized over the
    supplied bin range. A zero total count cannot be normalized.

    For redshift with reference PDZs, replace the indicator by the integrated
    normalized reference density :math:`p^j(z)`:

    .. math::

        N_b=\sum_{j\in\Phi_{\mathcal R_{\rm obs}}}
            w_{\rm rec}^j\int_{e_b}^{e_{b+1}}p^j(z)\,\mathrm dz.

    The implementation linearly interpolates Dataset's reference cumulative PDZs
    at bin edges and takes differences. Cumulative values outside ``PDZ_x`` are
    zero below the grid and one above it. PDZs affect only the final redshift
    histogram; they are not truncated to the observed selection interval and do
    not alter feature selection, matching or ``reference_weights``. Other
    properties always use point estimates. Test PDZs are ignored with a warning.

    By default, ``unmatched='drop'`` excludes unsupported positive-weight test
    objects with a warning. The reconstruction then describes the supported
    subset; ``diagnostics['weight_coverage']`` reports its fraction of selected
    test weight. Use ``unmatched='raise'`` to require support for every
    positive-weight selected test object. Zero supported weight always raises.
    Property values or PDZ mass outside the supplied histogram range are also
    excluded, with a warning; normalized output is conditional on that range.

    For either Dataset, its declared ``z_gal_string`` is resolved to the
    standardized ``'redshift'`` column before lookup. Other names are literal.
    With no declared redshift column, ``get_distribution('redshift', ...,
    use_pdzs=True)`` can still reconstruct supplied reference PDZs without point
    redshifts, provided the observed selection property is available in both
    samples. Original input PDZs are preserved; internal views disable them for
    observed selection and matching. Keep input datasets unchanged throughout
    this object's lifetime because indices and weights are cached.

    This class checks finite nonnegative input weights and ignores zero weights;
    Dataset may impose stricter validation when it is constructed. Feature
    selection preserves binary-selection metrics within its tolerance; it does
    not directly optimize distribution accuracy. The reference sample must
    adequately represent the target population at compatible colors. No
    histogram uncertainties or bootstrap realizations are computed.

    Examples
    --------
    Select photometric-redshift tomography and reconstruct reference spectroscopic
    redshifts. Here ``photoz`` is a numerical column in both datasets, while the
    reference Dataset designates its spectroscopic-redshift column through
    ``z_gal_string``::

        reconstruction = Reconstruction(
            reference_dataset=reference,
            test_dataset=test,
            selection_variable="photoz",
            selection_min=0.4,
            selection_max=0.6,
        )
        reconstruction.select_features_grid()
        reconstruction.process(cpu_number=4, unmatched="raise")
        density = reconstruction.get_distribution(
            "redshift", edges, use_pdzs=False
        )

    If the reference Dataset contains PDZs, use ``use_pdzs=True`` to transfer their
    bin integrals, or None (the default) to select that mode automatically for
    redshift. A different numerical property can reuse the same transferred
    weights, for example ``get_distribution('stellar_mass', mass_edges)``.
    """

    def __init__(
        self,
        reference_dataset,
        test_dataset,
        selection_variable,
        selection_min,
        selection_max,
        verbose=True,
    ):
        r"""
        Validate full parent samples and prepare observed-interval feature selection.

        See the class docstring for parameters and conventions. Validate selection
        values and weights in both samples, store selected row positions and weights,
        and construct a reference-only PointProbability for feature calibration.
        No feature calibration, test matching or weight transfer is performed here.

        Both samples must have positive weight inside and outside the observed
        interval. Test PDZs are ignored and produce a warning. The inputs are retained
        by reference and must not be modified after construction.
        """
        for name, ds in [("reference", reference_dataset), ("test", test_dataset)]:
            if not isinstance(ds, Dataset):
                raise TypeError(f"{name}_dataset must be a Dataset.")
        if (
            reference_dataset is test_dataset
            or reference_dataset.dataset is test_dataset.dataset
        ):
            raise ValueError(
                "Reconstruction requires separate reference and test samples. "
                "Do not pass the same Dataset or share its underlying DataFrame."
            )
        if not isinstance(selection_variable, str) or not selection_variable:
            raise TypeError("selection_variable must be a non-empty string.")

        lower, upper = float(selection_min), float(selection_max)
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("Selection bounds must be finite and min < max.")

        self.reference_dataset = reference_dataset
        self.test_dataset = test_dataset
        self.selection_variable = selection_variable
        self.selection_min, self.selection_max = lower, upper
        self.verbose = verbose
        self.colors = None
        self.discarded_colors = []
        self.reference_weights = None
        self.diagnostics = None

        self._ref_key = self._resolve(reference_dataset, selection_variable)
        ref_mask = self._selection_mask(reference_dataset, self._ref_key, "reference")
        test_key = self._resolve(test_dataset, selection_variable)
        test_mask = self._selection_mask(test_dataset, test_key, "test")
        self.reference_indices = np.flatnonzero(ref_mask)
        self.test_indices = np.flatnonzero(test_mask)
        self._ref_weights = self._weights(reference_dataset).copy()
        self._test_weights = self._weights(test_dataset)[test_mask].copy()

        # Use the generic additional_rules API. An internal alias also supports
        # Dataset's reserved "redshift" column without special selection logic.
        self._selection_key = "_reconstruction_selection_property"
        while self._selection_key in reference_dataset.dataset.columns:
            self._selection_key += "_"

        self._selection = SelectionDefinition(
            additional_rules=[
                (
                    self._selection_key,
                    lambda values: (values > lower) & (values <= upper),
                )
            ]
        )

        # feature_reference is used for feature calibration only.
        # _selection_reference copies _selection_key in a column.
        # SelectionDefinition.additional_rules will base its computations on this variable.
        # PDZs are disabled ONLY in this internal view: sample selection and
        # feature calibration use the observed selection variable.
        # Original reference CDZs remain available for get_distribution().
        feature_reference = self._selection_reference(reference_dataset)

        # For feature calibration only: this object never performs test matching
        self._feature_point = PointProbability(
            selection_definition=self._selection,
            reference_dataset=feature_reference,
            verbose=verbose,
        )

        if test_dataset.CDZs is not None:
            warnings.warn(
                "Test-sample PDZs are ignored. Reconstruction transfers "
                "reference point estimates or reference PDZs to the test sample.",
                stacklevel=2,
            )
        if verbose:
            print(
                f"Reconstruction selection: {lower} < {selection_variable} <= {upper}. "
                f"Reference: {ref_mask.sum()}/{len(ref_mask)}; "
                f"test: {test_mask.sum()}/{len(test_mask)}.\n"
                "Feature selection uses the full reference. Matching uses selected "
                "subsets.\n"
            )

    def _selection_reference(self, ds):
        r"""
        Build an internal reference view for point-based interval membership.

        Shallow-copy the Dataset and its DataFrame, add the resolved observed
        selection column under the private alias used by additional_rules, and
        clear the redshift designation and PDZ arrays on the view. Original PDZs
        remain attached to the input Dataset for final redshift reconstruction.
        """
        result = copy(ds)
        result.dataset = ds.dataset.copy(deep=False)
        result.dataset[self._selection_key] = ds.dataset[self._ref_key].to_numpy()
        result.z_gal_string = None
        result.CDZs = None
        result.PDZ_x = None
        return result

    @staticmethod
    def _resolve(ds, variable):
        r"""
        Resolve a property name to the stored Dataset column.

        A name matching the Dataset's non-None z_gal_string maps to ``'redshift'``,
        which takes precedence over any retained original column. All other names
        are literal. Raise KeyError if the resolved column is absent.
        """
        requested = variable
        if ds.z_gal_string is not None and variable == ds.z_gal_string:
            variable = "redshift"
        if variable in ds.dataset.columns:
            return variable
        raise KeyError(
            f"Column {requested!r} (resolved to {variable!r}) "
            "is absent from a required dataset."
        )

    @staticmethod
    def _values(ds, key):
        r"""
        Return numerical property values and a missing-value mask.

        Convert the resolved column to a float array. Flag nonfinite values and
        entries matching Dataset.missing_data; return ``(values, invalid)`` aligned
        with the Dataset rows. Non-numerical columns cannot be reconstructed.
        """
        column = ds.dataset[key]
        values = column.to_numpy(dtype=float, na_value=np.nan)
        sentinels = ds.missing_data
        if sentinels is None:
            sentinels = []
        elif np.isscalar(sentinels):
            sentinels = [sentinels]
        invalid = ~np.isfinite(values) | column.isin(sentinels).to_numpy()
        return values, invalid

    @staticmethod
    def _weights(ds):
        r"""
        Return finite nonnegative Dataset weights, checking their total.

        Raise ValueError for negative or nonfinite entries, or a nonfinite total.
        Zero weights are accepted here and contribute no transferred weight.
        """
        weights = ds.dataset["weights"].to_numpy(dtype=float, na_value=np.nan)
        if np.any(~np.isfinite(weights) | (weights < 0)):
            raise ValueError("Dataset weights must be finite and nonnegative.")
        if not np.isfinite(weights.sum()):
            raise ValueError("Total weight overflows; rescale input weights.")
        return weights

    def _selection_mask(self, ds, key, name):
        r"""
        Validate and select ``selection_min < s <= selection_max``.

        Use the resolved numerical column key. Reject any missing or nonfinite
        selection value, and require positive-weight objects both inside and
        outside the interval. These checks apply separately to reference and test
        parent samples. Return a Boolean mask aligned with all input rows; selection
        does not itself remove zero-weight objects.
        """
        values, invalid = self._values(ds, key)
        if invalid.any():
            raise ValueError(
                f"{name}: {invalid.sum()} invalid selection values in "
                f"{key!r}. Remove missing selection values first."
            )
        mask = (values > self.selection_min) & (values <= self.selection_max)
        positive = self._weights(ds) > 0
        if not np.any(mask & positive):
            raise ValueError(f"{name}: no positive-weight objects inside selection.")
        if not np.any(~mask & positive):
            raise ValueError(
                f"{name}: all positive-weight objects are inside the selection. "
                "Pass the FULL parent dataset, not the preselected sample. "
                "Feature selection requires reference objects both inside and "
                "outside the interval; both inputs are checked for preselection."
            )
        return mask

    def select_features_grid(
        self,
        tolerance=0.1,
        purity_min=0.9,
        grid_precision=0.15,
        step=0.01,
        retry_calibration=True,
    ):
        r"""
        Select colors for observed-interval membership on the full reference sample.

        Delegate grid calibration and backward elimination to PointProbability.
        Reference prior weights enter the calibration metrics; test data and test
        weights do not. The target is always
        ``selection_min < selection_variable <= selection_max``, evaluated from
        observed point values. Reference PDZs are disabled for this step.

        Parameters
        ----------
        tolerance : float, default=0.1
            Maximum absolute relative change in purity and completeness from the
            fixed measured initial calibration values during backward elimination.
        purity_min : float, default=0.9
            Requested minimum global purity for the initial grid calibration.
            The measured initial purity may be higher. Greedy elimination compares
            with that measured baseline and can yield a final purity below the
            requested minimum within the allowed tolerance.
        grid_precision : float, default=0.15
            Color-grid resolution passed to GridClassification.
        step : float, default=0.01
            Reduction step for the bin-probability threshold scanned by grid
            calibration. This is not a histogram bin width or a purity decrement.
        retry_calibration : bool, default=True
            Retry the recognized ``purity_not_crossed`` failure using the reported
            calibration purity plus 0.001, only when the proposed threshold is
            finite, larger than the current threshold and below one. A successful
            adjusted calibration is reported by PointProbability through a warning.

        Returns
        -------
        list of tuple(str, str)
            Retained colors, also stored in ``colors``. Removed colors are stored in
            ``discarded_colors``.

        Raises
        ------
        ValueError, TypeError
            Propagated for invalid inputs or unsuccessful grid calibration. A
            ``no_accepted_selection`` failure is not recovered by borrowing features.

        Notes
        -----
        The selected set is calibrated once for the observed interval, not once per
        bin of the property reconstructed later. Unlike ProbabilityFunction, this
        class neither switches to a complementary criterion nor copies colors from
        another threshold. Bootstrap estimation is disabled in the delegated grid
        calibration; its default alpha=1 disables significance filtering.

        Each elimination trial removes all active grids involving one candidate
        color. Remaining fixed masks are combined through OR. Only the candidate
        with the highest remaining completeness is tested against both fixed
        baseline tolerances. Removal stops if that candidate fails or two colors
        remain. See PointProbability.select_features_grid for the implementation.

        This call clears any existing transferred weights and diagnostics before
        calibration. Call process afterwards, including after repeating feature
        selection on an already processed object.
        """
        self.reference_weights = None
        self.diagnostics = None
        self.colors = None
        self._feature_point.select_features_grid(
            tolerance=tolerance,
            purity_min=purity_min,
            grid_precision=grid_precision,
            step=step,
            retry_calibration=retry_calibration,
        )
        self.colors = list(self._feature_point.colors)
        self.discarded_colors = list(self._feature_point.discarded_colors)
        return self.colors

    @staticmethod
    def _subset(ds, indices):
        r"""
        Copy selected Dataset rows into a view used for matching.

        Select positional indices, copy the DataFrame and reset its row labels.
        Disable CDZs and PDZ_x only on the copied Dataset; original PDZ arrays remain
        available through reference_dataset for the final distribution estimate.
        """
        result = copy(ds)
        result.dataset = ds.dataset.iloc[indices].copy().reset_index(drop=True)
        result.CDZs = None
        result.PDZ_x = None
        return result

    def process(
        self, min_colors_used=2, cpu_number=None, block_size=256, unmatched="drop"
    ):
        r"""
        Match selected samples and transfer supported test weight to reference objects.

        For each supported selected test object, normalize the positive reference
        prior weights over its color-compatible set and multiply by that test
        object's weight. Accumulate these contributions in reference_weights.
        The class docstring defines :math:`\Sigma_i` and :math:`w_{\rm rec}^j`.
        No per-object membership probabilities or reconstructed histograms are
        computed by this method.

        Parameters
        ----------
        min_colors_used : int, default=2
            Minimum number of retained colors with finite test values and errors.
            Must be positive and no larger than the retained color count. A test
            object with fewer usable colors is unsupported.
        cpu_number : int or None, default=None
            Maximum worker threads over blocks of selected test objects. None uses
            ``max(cpu_count() - 2, 1)``; explicit positive integers are capped at that
            limit. Use 1 for sequential processing. No process-based spline fitting
            is involved in Reconstruction.
        block_size : int, default=256
            Positive number of selected test objects per task. Each task accumulates
            a vector with one entry per selected reference object. Bounded batches
            limit the number of these vectors retained simultaneously.
        unmatched : {'raise', 'drop'}, default='drop'
            Policy for positive-weight selected test objects with insufficient
            usable colors or no positive-weight compatible reference. 'drop' warns
            and reconstructs only the supported subset. 'raise' rejects any such
            object. Zero supported test weight raises under either policy.

        Returns
        -------
        None
            On success, store transferred weights aligned with the full reference
            DataFrame in reference_weights. Outside-selection entries are zero.
            Populate diagnostics with:

            test_indices : numpy.ndarray of int
                Selected row positions in the original test DataFrame.
            supported : numpy.ndarray of bool
                Supported positive-weight test objects.
            n_matches : numpy.ndarray of int
                Number of positive-weight compatible reference objects per test.
            n_effective : numpy.ndarray of float
                Effective match count, :math:`1/\sum_{j\in\Sigma_i}\alpha_{ij}^2`,
                where :math:`\alpha_{ij}=w_{\rm prior}^j/
                \sum_{k\in\Sigma_i}w_{\rm prior}^k`.
            n_colors_used : numpy.ndarray of int
                Number of finite retained test colors and errors. Zero-weight test
                objects are skipped and retain zero in this array.
            selected_test_weight, supported_test_weight : float
                Total input weight of the selected test subset and its supported part.
            weight_coverage : float
                supported_test_weight / selected_test_weight.

            All per-object diagnostic arrays follow test_indices. Unsupported
            objects have zero matches and effective counts; zero-weight objects are
            not flagged as supported. Diagnostics are also available when a run
            reaches the unsupported-object policy check and raises there.

        Raises
        ------
        RuntimeError
            If colors have not been selected, or transferred weights fail the
            supported-test-weight conservation check.
        ValueError
            For invalid settings, finite negative magnitude errors in retained
            bands of either selected sample, or a support failure under the chosen
            policy. Lower-level input errors may also propagate.

        Notes
        -----
        Only objects satisfying the observed interval participate in matching.
        Nonfinite color values or errors are made unavailable; zero errors are
        allowed. Reference and test errors are combined in quadrature, since the
        samples are distinct. At every usable test color, a reference analogue
        must have usable photometry and satisfy the matching tolerance.

        Matching reuses PointProbability routines, including the compiled matching
        kernel, but does not call PointProbability.process. Reference/test PDZs are
        disabled in these internal matching views. Neither a redshift-bin probability
        nor a further reference-weight factor is applied to the transferred weights.

        After parameter validation, previous transferred weights and diagnostics are
        cleared. A support failure does not commit new reference_weights. The method
        can be rerun; changing colors requires a new run before get_distribution.
        Keep the input Dataset contents and row ordering unchanged.
        """
        # Initial checks
        if self.colors is None:
            raise RuntimeError("Run select_features_grid() before process().")
        for name, value in [
            ("min_colors_used", min_colors_used),
            ("block_size", block_size),
            ("cpu_number", cpu_number),
        ]:
            if name == "cpu_number" and value is None:
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if min_colors_used > len(self.colors):
            raise ValueError("min_colors_used exceeds the number of selected colors.")
        if unmatched not in ("raise", "drop"):
            raise ValueError("unmatched must be 'raise' or 'drop'.")

        # Initialize the reference and test subsamples, selected on selection_variable
        self.reference_weights = None
        self.diagnostics = None
        ref = self._selection_reference(
            self._subset(self.reference_dataset, self.reference_indices)
        )
        test = self._subset(self.test_dataset, self.test_indices)

        # Suppress only PointProbability's warning about its own
        # probability calculation, which we do not call.
        # Then, set all internal variables of PointProbability.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Prior weights were set in test_dataset, but they are not used by the PointProbability class.",
                category=UserWarning,
            )
            # Both selected samples retain their <band>_error columns.
            point = PointProbability(self._selection, ref, test, verbose=False)

        point.colors = list(self.colors)
        point._set_dictionaries()

        for matrices in (point.reference_dict, point.test_dict):
            # Infinities and invalid errors must never provide a matching window
            colors, errors = matrices["colors"], matrices["errors"]
            invalid = ~np.isfinite(colors) | ~np.isfinite(errors)
            errors[invalid] = np.nan
            colors[invalid] = np.nan

        # Reject negative band errors before squaring in the shared color builder
        for ds in (ref, test):
            for band in {b for color in self.colors for b in color}:
                err = ds.dataset[band + "_error"].to_numpy(dtype=float)
                if np.any(np.isfinite(err) & (err < 0)):
                    raise ValueError(f"Negative magnitude errors in {band}_error.")

        # Initialize
        rw = self._ref_weights[self.reference_indices]  # reference prior weights
        nt, nr = len(self.test_indices), len(self.reference_indices)
        counts = np.zeros(nt, dtype=int)
        neff = np.zeros(nt)
        ncolors = np.zeros(nt, dtype=int)
        supported = np.zeros(nt, dtype=bool)

        point._matching_indices(
            0
        )  # Compile/load the Numba specialization before starting workers

        # The run_block function outputs contributions, having dimension nr.
        # It runs over blocks of test objects. For each test i,
        # the code skips it if it has null weight or not enough usable colors.
        # Otherwise, it finds matches with point._matching_indices(i).
        # Then, alpha contains the normalized prior weights of the matched
        # reference objects with positive weight:
        # alpha[j] = w_ref[j] / sum(w_ref[k] for k in matches).
        # The test object's prior weight is distributed among these references,
        # adding w_test[i] * alpha[j] to each reference contribution.
        # Thus, each supported test object transfers its entire prior weight.
        def run_block(start):
            contributions = np.zeros(nr)
            for i in range(start, min(start + block_size, nt)):
                if self._test_weights[i] == 0:
                    continue
                usable = np.isfinite(point.test_dict["colors"][:, i]) & np.isfinite(
                    point.test_dict["errors"][:, i]
                )
                ncolors[i] = usable.sum()
                if ncolors[i] < min_colors_used:
                    continue
                indices = point._matching_indices(i)
                indices = indices[rw[indices] > 0]
                counts[i] = len(indices)
                if not len(indices):
                    continue
                # Normalize the matched reference weights so they sum to one
                matched_weights = rw[indices]
                alpha = matched_weights / matched_weights.sum()
                neff[i] = 1.0 / np.dot(alpha, alpha)
                supported[i] = True
                contributions[indices] += self._test_weights[i] * alpha
            return contributions

        workers = min(
            cpu_number or max(mp.cpu_count() - 2, 1), max(mp.cpu_count() - 2, 1)
        )
        starts = list(range(0, nt, block_size))
        total = np.zeros(nr)

        with tqdm(
            total=nt,
            desc="Reconstructing reference contributions",
            disable=not self.verbose,
        ) as progress:
            if workers == 1:
                for start in starts:
                    total += run_block(start)
                    progress.update(min(block_size, nt - start))
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    # Bounded batches prevent retaining one reference-sized
                    # accumulator for every queued test block
                    for offset in range(0, len(starts), workers):
                        batch = starts[offset : offset + workers]
                        for start, contribution in zip(
                            batch, pool.map(run_block, batch)
                        ):
                            total += contribution
                            progress.update(min(block_size, nt - start))

        missing = (~supported) & (self._test_weights > 0)
        selected_weight = self._test_weights.sum()
        supported_weight = self._test_weights[supported].sum()

        self.diagnostics = dict(
            test_indices=self.test_indices.copy(),
            supported=supported,
            n_matches=counts,
            n_effective=neff,
            n_colors_used=ncolors,
            selected_test_weight=float(selected_weight),
            supported_test_weight=float(supported_weight),
            weight_coverage=float(supported_weight / selected_weight),
        )

        if missing.any():
            message = (
                f"{missing.sum()} positive-weight selected test objects lack "
                f"usable analogues. Supported test-weight fraction: "
                f"{supported_weight/selected_weight:.6f}. "
                f"First unsupported input positions: "
                f"{self.test_indices[missing][:10].tolist()}."
            )
            if unmatched == "raise" or supported_weight <= 0:
                raise ValueError(message + " No distribution was committed.")
            warnings.warn(
                message + " Reconstruction describes ONLY the supported subset.",
                stacklevel=2,
            )
        if not np.isclose(total.sum(), supported_weight, rtol=1e-10, atol=0):
            raise RuntimeError(
                "Transferred weights do not conserve supported test weight."
            )
        self.reference_weights = np.zeros(len(self.reference_dataset.dataset))
        self.reference_weights[self.reference_indices] = total
        self._processed_colors = list(self.colors)
        if self.verbose:
            print(f"Matching complete.")

    def get_distribution(self, variable, bin_edges, normalize=True, use_pdzs=None):
        r"""
        Bin transferred reference weights into a selected-test-sample distribution.

        Use reference_weights from a successful process call. For point estimates,

        .. math::

            N_b=\sum_{j\in\Phi_{\mathcal R_{\rm obs}}}
                w_{\rm rec}^j\,\delta^j(x\in B_b).

        For redshift with normalized reference PDZs :math:`p^j(z)`, instead compute

        .. math::

            N_b=\sum_{j\in\Phi_{\mathcal R_{\rm obs}}}
                w_{\rm rec}^j\int_{e_b}^{e_{b+1}}p^j(z)\,\mathrm dz.

        No extra prior or test weight is applied: both are already included in
        :math:`w_{\rm rec}^j`. The observed interval, selected colors and matching
        sets are unchanged. This method can be called repeatedly for different
        reference properties, bin edges and normalization choices.

        Parameters
        ----------
        variable : str
            Numerical reference property :math:`x`. It is not required in the test
            sample. It should differ from selection_variable, although this is not
            enforced. Use 'redshift' or the reference Dataset's declared z_gal_string
            alias for redshift reconstruction; other names refer to point columns.
        bin_edges : array-like of float
            Finite, strictly increasing one-dimensional edges, with at least two
            entries. Unequal widths are supported. Point-estimate bins are
            ``[left, right)``, except the last, which also includes its right edge.
            Edges need not fall within the observed sample-selection interval.
        normalize : bool, default=True
            Return the density :math:`n_b=N_b/[(e_{b+1}-e_b)\sum_cN_c]` if True,
            or unnormalized weighted counts :math:`N_b` if False. With unit test
            weights, each supported test object contributes one count in total
            when bins cover all transferred values or PDZ probability mass.
        use_pdzs : bool or None, default=None
            None automatically uses reference PDZs for redshift when available,
            otherwise point estimates. True requires reference PDZs and a redshift
            variable. False forces point estimates and warns when available
            redshift PDZs are bypassed. PDFs for arbitrary other properties are
            not supported. Test PDZs are never used.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(bin_edges) - 1,)``; edges are not returned. Density units
            are inverse units of variable. Unnormalized counts have test-weight
            units. Normalized output integrates to one over the supplied range.

        Raises
        ------
        RuntimeError
            If process has not succeeded or the color list changed afterwards.
        TypeError, KeyError
            For an invalid variable or use_pdzs argument, or a missing required
            reference column.
        ValueError
            For invalid bin edges, unavailable or invalid requested PDZs, PDZ use
            for another property, missing redshift designation in point-redshift
            mode, invalid point values in contributing references, or zero weight
            inside the range when normalize=True.

        Notes
        -----
        The declared reference z_gal_string resolves to the standardized 'redshift'
        column even if its original column remains present. With z_gal_string=None,
        use 'redshift' to access supplied reference PDZs without point redshifts;
        point-redshift reconstruction requires a declared redshift column.

        Only reference rows with positive transferred weight are used. Invalid
        point values in those rows cause an error rather than silent removal.
        PDZ mode validates the contributing cumulative distributions and integrates
        by linear interpolation at bin edges; see _pdz_histogram.

        Values or PDZ mass outside the bin range are omitted. A lost-weight fraction
        above 1e-12 triggers a warning. Normalization is conditional on the retained
        range and, if unmatched='drop' excluded test objects, on the supported test
        subset. With normalize=False, an empty represented range returns zero counts.
        No uncertainty estimates are returned.
        """
        # Initial checks
        if self.reference_weights is None:
            raise RuntimeError("Run process() successfully before get_distribution().")
        if list(self.colors) != self._processed_colors:
            raise RuntimeError("Colors changed; run process() again.")
        edges = np.asarray(bin_edges, dtype=float)
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.isfinite(edges).all()
            or not np.all(np.diff(edges) > 0)
        ):
            raise ValueError("bin_edges must be finite and strictly increasing.")
        if not isinstance(variable, str) or not variable:
            raise TypeError("variable must be a non-empty column-name string.")
        if use_pdzs is not None and not isinstance(use_pdzs, (bool, np.bool_)):
            raise TypeError("use_pdzs must be a boolean or None.")

        reference = self.reference_dataset

        # Dataset attaches PDZs to redshift, not to every numerical property
        is_redshift = variable == "redshift" or (
            reference.z_gal_string is not None and variable == reference.z_gal_string
        )
        has_pdzs = reference.CDZs is not None
        pdz_mode = (is_redshift and has_pdzs) if use_pdzs is None else bool(use_pdzs)

        if pdz_mode:
            if not is_redshift:
                raise ValueError(
                    "Dataset PDZs describe redshift. Use variable='redshift' "
                    "or the reference Dataset's z_gal_string alias; they cannot "
                    "be applied to an unrelated property."
                )
            if not has_pdzs:
                raise ValueError("use_pdzs=True requires reference PDZs.")
            hist, lost_fraction = self._pdz_histogram(edges)
        else:
            if is_redshift and reference.z_gal_string is None:
                raise ValueError(
                    "Point-estimate redshift reconstruction requires "
                    "reference_dataset.z_gal_string to identify the "
                    "redshift column. To use reference PDZs instead, "
                    "set use_pdzs=None or True and provide PDZs."
                )
            key = self._resolve(reference, variable)
            values, invalid = self._values(reference, key)
            active = self.reference_weights > 0
            if np.any(active & invalid):
                raise ValueError(
                    f"{np.count_nonzero(active & invalid)} contributing "
                    f"reference objects have invalid {variable!r}."
                )
            outside = active & ((values < edges[0]) | (values > edges[-1]))
            lost_fraction = (
                self.reference_weights[outside].sum() / self.reference_weights.sum()
            )
            hist = np.histogram(
                values[active], bins=edges, weights=self.reference_weights[active]
            )[0]
            if is_redshift and has_pdzs and use_pdzs is not None:
                warnings.warn(
                    "use_pdzs=False: reference PDZs are ignored for this "
                    "distribution; using redshift point estimates instead.",
                    stacklevel=2,
                )

        if lost_fraction > 1e-12:
            warnings.warn(
                f"{lost_fraction:.6%} of transferred weight lies outside the "
                "represented bin range. Normalized output is conditional on "
                "the retained range.",
                stacklevel=2,
            )
        if self.verbose:
            method = (
                "reference PDZ integrals" if pdz_mode else "reference point estimates"
            )
            print(f"Reconstructing {variable!r} using {method}.")
        if normalize:
            if hist.sum() <= 0:
                raise ValueError("No transferred weight inside bin_edges.")
            hist = hist / hist.sum() / np.diff(edges)
        return hist

    def _pdz_histogram(self, edges):
        r"""
        Compute transferred-weight counts from normalized reference cumulative PDZs.

        For each positive-weight contributing reference, linearly interpolate its
        cumulative redshift distribution :math:`C_{\rm ref}^j` at the bin edges:

        .. math::

            N_b=\sum_j w_{\rm rec}^j
                [C_{\rm ref}^j(e_{b+1})-C_{\rm ref}^j(e_b)].

        This numerically represents the bin integral of :math:`p^j(z)` in the
        class docstring. Weights are scaled by their total before accumulation and
        rescaled afterwards. Bin masses are clipped to [0, 1] for numerical residuals.

        Parameters
        ----------
        edges : numpy.ndarray
            Finite, strictly increasing redshift edges, already validated by
            get_distribution.

        Returns
        -------
        counts : numpy.ndarray
            Unnormalized weighted counts of shape ``(len(edges) - 1,)``. With full
            PDZ-grid coverage, their sum equals total transferred reference weight.
        lost_fraction : float
            Transferred-weight fraction of probability mass outside the bin range,
            not the fraction of objects outside it.

        Raises
        ------
        ValueError
            For an invalid PDZ grid, a cumulative-array shape inconsistent with the
            full reference, or invalid contributing cumulative distributions.
            Rows must be finite, nondecreasing and normalized from zero to one over
            PDZ_x, within a tolerance of 1e-10. Excess total retained mass also raises.

        Notes
        -----
        Requires a successful process call and positive total transferred weight.
        PDZ arrays retain the full input-reference row ordering. Only rows with
        positive reference_weights are validated and used. Below and above PDZ_x,
        interpolated cumulative values are zero and one. The supplied grid defines
        normalized PDZ support; unrepresented probability outside it is unknown.
        No matching, feature selection, density normalization or warning emission
        is performed here. get_distribution handles normalization and lost weight.
        """
        reference = self.reference_dataset
        grid = np.asarray(reference.PDZ_x, dtype=float)
        cdfs = np.asarray(reference.CDZs, dtype=float)

        # Initial checks
        if (
            grid.ndim != 1
            or grid.size < 2
            or not np.isfinite(grid).all()
            or not np.all(np.diff(grid) > 0)
        ):
            raise ValueError("Reference PDZ_x must be finite and strictly increasing.")
        if cdfs.shape != (len(reference.dataset), grid.size):
            raise ValueError(
                "Reference CDZs must align with the full reference and PDZ_x."
            )

        active = self.reference_weights > 0
        rows = cdfs[active]
        tolerance = 1e-10
        if (
            not np.isfinite(rows).all()
            or np.any(rows < -tolerance)
            or np.any(rows > 1 + tolerance)
            or np.any(np.diff(rows, axis=1) < -tolerance)
            or not np.allclose(rows[:, 0], 0, rtol=0, atol=tolerance)
            or not np.allclose(rows[:, -1], 1, rtol=0, atol=tolerance)
        ):
            raise ValueError(
                "Contributing reference CDZs must be finite, nondecreasing, "
                "and normalized from zero to one over PDZ_x."
            )

        # Evaluate each reference CDF at the requested bin edges.
        # Outside PDZ_x, use the endpoints of the normalized distribution.
        edge_cdfs = np.array(
            [np.interp(edges, grid, cdf, left=0.0, right=1.0) for cdf in rows]
        )

        # Normalize weights before multiplication to avoid overflow. The bin
        # probabilities equal the weighted average of PointProbability values.
        weights = self.reference_weights[active]
        total_weight = weights.sum()
        probabilities = np.clip(np.diff(edge_cdfs, axis=1), 0.0, 1.0)
        mass = (weights / total_weight) @ probabilities
        retained = mass.sum()
        if retained > 1 + tolerance:
            raise ValueError("Reference PDZ bin probabilities exceed unit integral.")

        return mass * total_weight, max(0.0, 1.0 - retained)
