"""
Transfer property distributions from a reference to a selected test sample.
"""

from copy import copy
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import warnings

import numpy as np
from tqdm import tqdm

from ..data.dataset import Dataset
from ..data.selection_definition import SelectionDefinition
from ..probability.point_probability import PointProbability


class Reconstruction:
    r"""
    Reconstruct properties of a test sample using photometric features.

    Parameters
    ----------
    reference_dataset, test_dataset : Dataset
        Distinct reference and test datasets, BEFORE the interval selection.
        Passing the same Dataset instance or sharing the same underlying
        DataFrame raises a ValueError. Both samples must
        contain positive-weight objects inside AND outside the interval.
        Both samples must provide photometric magnitudes and their errors in
        columns named ``<band>`` and ``<band>_error``, with the same candidate
        colors. Errors from the test sample are used in photometric matching,
        together with the reference errors.
        Input ``weights`` have different roles: reference weights set the
        relative contributions of analogues, while test weights set how much
        each test object contributes to the reconstructed population.
    selection_variable : str
        Numerical property defining the selection, available in both samples.
        This can differ from the property reconstructed by
        :meth:`~fomoz.reconstruction.reconstruction.Reconstruction.get_distribution`,
        which is required only in the reference sample.
        In each Dataset, its non-None ``z_gal_string`` is resolved to
        ``"redshift"`` before looking up a column, even if the original
        column is retained. The standardized column takes precedence.
        Other names are read literally. With ``z_gal_string=None``, no
        original-name alias is inferred.

        Selection always uses observed numerical values, never PDZ integrals.
        Ensure the requested property has the same physical meaning in both
        samples; their designated redshift columns need not represent the
        same estimator. For example, select on a separate photometric-redshift
        column and reconstruct spectroscopic redshift from the reference.
    selection_min, selection_max : float
        Selection is ``selection_min < value <= selection_max`` throughout,
        including feature calibration. These limits do NOT cut the property
        passed to :meth:`~fomoz.reconstruction.reconstruction.Reconstruction.get_distribution`.
    verbose : bool, default=True
        Print selection counts, the reconstruction method, and matching
        progress. Validation errors and warnings are not suppressed.

    Attributes
    ----------
    colors : list of tuple(str, str) or None
        Colors retained by
        :meth:`~fomoz.reconstruction.reconstruction.Reconstruction.select_features_grid`;
        ``None`` before calibration.
    discarded_colors : list of tuple(str, str)
        Candidate colors removed by feature selection.
    reference_indices, test_indices : numpy.ndarray
        Positional indices of objects inside the selection in each original
        :class:`~fomoz.data.dataset.Dataset` DataFrame. These are row positions,
        not DataFrame index labels.
    reference_weights : numpy.ndarray or None
        Transferred test weights aligned to the FULL reference DataFrame.
        Entries outside the selection are zero. Available after successful
        :meth:`~fomoz.reconstruction.reconstruction.Reconstruction.process`;
        their sum equals the weight of supported selected test objects.
    diagnostics : dict or None
        Matching diagnostics set by
        :meth:`~fomoz.reconstruction.reconstruction.Reconstruction.process`.

    Notes
    -----
    The constructor validates the full parent samples, records the interval
    masks, and prepares an internal
    :class:`~fomoz.probability.point_probability.PointProbability` for feature
    selection. It does not yet perform feature calibration or matching.
    The reconstruction then proceeds as follows:

    1. Call :meth:`~fomoz.reconstruction.reconstruction.Reconstruction.select_features_grid`.
       The internal ``PointProbability`` uses the FULL reference sample to
       identify colors that distinguish objects inside the selection interval
       from those outside it. No external test sample is needed for this step.
    2. Call :meth:`~fomoz.reconstruction.reconstruction.Reconstruction.process`.
       Both parent samples are restricted internally to the same interval in
       ``selection_variable``. A SECOND internal ``PointProbability`` receives
       these selected reference and test samples and the previously selected
       colors. Its matching routines use the photometric errors of the test
       objects as well as those of their candidate reference analogues.
    3. For each supported test object, normalize the positive reference weights
       over its compatible analogues. Assign these analogues a total
       contribution equal to that test object's weight, then sum contributions
       from all supported test objects. No membership probability is used as
       an additional weight.
    4. Call :meth:`~fomoz.reconstruction.reconstruction.Reconstruction.get_distribution`
       with a reference-property name and histogram edges. It transfers point
       estimates, or reference PDZ bin integrals when reconstructing redshift,
       using the accumulated contributions. The property may be
       unavailable in the test sample and need not equal ``selection_variable``.
       Its values are NOT restricted to the interval used for sample selection.

    We adopt the notation of
    :class:`~fomoz.probability.point_probability.PointProbability`.
    Let :math:`\Phi` denote the full reference sample and :math:`\mathcal{R}`
    the interval selection. Matching uses its selected subset
    :math:`\Phi_{\mathcal{R}}`. Feature selection on the full reference retains
    :math:`X_{\mathrm{sel}}`; for test object :math:`i`, :math:`\chi_i`
    contains the retained colors with usable measurements and errors.
    Compatible reference objects form the set

    .. math::

        \Sigma_i = \left\{
            j \in \Phi_{\mathcal{R}} :
            |c_{i,k} - c_{j,k}| \leq \epsilon_{ij,k}
            \quad \forall k \in \chi_i
        \right\},
        \qquad
        \epsilon_{ij,k} =
        \sqrt{\sigma_{i,k}^{2} + \sigma_{j,k}^{2}}.

    The indices :math:`i` and :math:`j` denote test and reference objects,
    respectively. At least ``min_colors_used`` usable test colors are required.
    For :math:`c_k=m_a-m_b`, the color uncertainty is

    .. math::

        \sigma_k = \sqrt{\sigma_{m_a}^{2} + \sigma_{m_b}^{2}}.

    Errors are assumed uncorrelated between bands and between samples.
    Covariance between colors sharing a band is not modeled. Reference and
    test must be separate samples. The constructor rejects a shared Dataset
    or DataFrame instance but cannot identify duplicated catalogues passed as
    independent copies or overlapping objects in distinct catalogues.

    Let :math:`x` be the reconstructed property,
    and :math:`B_b=[x_b,x_{b+1})` a histogram interval. The empirical probability
    assigned to test object :math:`i` has the same form as in PointProbability:

    .. math::

        p_i(B_b) =
        \frac{\sum_{j\in\Sigma_i} w_{\mathrm{sel}}^{j}(B_b)}
             {\sum_{j\in\Sigma_i} w_{\mathrm{prior}}^{j}},
        \qquad
        w_{\mathrm{sel}}^{j}(B_b) =
        w_{\mathrm{prior}}^{j}\,q_j(B_b).

    With point estimates, :math:`q_j(B_b)=\mathbf{1}(x_j\in B_b)`.
    For a redshift reconstruction using reference PDZs instead,

    .. math::

        q_j(B_b)=\int_{z_b}^{z_{b+1}}P_j(z)\,\mathrm{d}z
                =F_j(z_{b+1})-F_j(z_b),

    where :math:`F_j` is the normalized cumulative PDZ stored by Dataset.
    Thus PDZs replace binary histogram-bin membership, not the reference prior
    weights. They are not truncated to the sample-selection interval.

    Here :math:`w_{\mathrm{prior}}^{j}` is the reference ``weights`` entry,
    as in PointProbability. The event :math:`x\in B_b` describes a histogram
    bin; it is NOT the initial sample selection :math:`\mathcal{R}`. The same
    selected colors and matching sets are used for all histogram intervals.
    These probabilities are implicit: no new PointProbability calibration is
    performed for each interval. For point estimates the final histogram bin
    includes its right edge, following NumPy's convention. PDZ integration uses
    the same discrete CDF lookup as SelectionDefinition in PointProbability:
    each edge is mapped to the first PDZ_x value at or above it, with clipping
    to the grid endpoints. Use histogram edges on PDZ_x for exact bin alignment.

    Denote the supported, positive-weight selected test sample by
    :math:`\mathcal{T}`. Its input weights are written
    :math:`w_{\mathrm{prior}}^{i}`: these are the test Dataset's ``weights``
    entries, not estimated membership probabilities. For bins covering all
    transferred property values, the reconstructed density is

    .. math::

        \widehat p_b =
        \frac{\sum_{i\in\mathcal{T}}
                    w_{\mathrm{prior}}^{i}\,p_i(B_b)}
             {\Delta x_b\sum_{i\in\mathcal{T}} w_{\mathrm{prior}}^{i}},
        \qquad \Delta x_b=x_{b+1}-x_b.

    The implementation computes the equivalent accumulated reference weights
    directly, without storing the probabilities for every object and bin:

    .. math::

        w_{\mathrm{rec}}^{j} =
        \sum_{i\in\mathcal{T}:\,j\in\Sigma_i}
        w_{\mathrm{prior}}^{i}
        \frac{w_{\mathrm{prior}}^{j}}
             {\sum_{\ell\in\Sigma_i}w_{\mathrm{prior}}^{\ell}}.

    These weights are stored in ``reference_weights``. They already contain
    the locally normalized reference weights and the test weights: neither
    an additional prior-weight factor nor :math:`p_i(\mathcal{R})` is applied.
    Unnormalized bin counts sum
    :math:`w_{\mathrm{rec}}^{j}q_j(B_b)` over the reference.
    Normalized output divides by the total transferred weight INSIDE the
    supplied histogram range and by each bin width. If that range excludes
    transferred values, the output is conditional on the retained range.

    By default (``unmatched='drop'``), unsupported positive-weight selected
    test objects are excluded with a warning. Then :math:`\mathcal{T}` and the
    distribution describe only the supported subset. ``unmatched='raise'``
    instead requires full support; zero supported test weight always raises.

    Reference/test weights must be finite and nonnegative; zero weights do not
    contribute. Sample selection and feature calibration ALWAYS use the
    point values of selection_variable. Reference PDZs, when used, replace
    only the redshift values in the final distribution. They do not change
    feature selection, matching, or reference_weights. The original reference
    CDZs remain stored in the input Dataset even though the internal matching
    views disable them. Test PDZs are unused and trigger a constructor warning.
    Photometric errors remain active in matching in all modes.

    Feature selection optimizes interval membership, not accuracy of the full
    reconstructed distribution. Representativity and transfer bias need external
    validation. No uncertainty on the reconstructed histogram is estimated.

    Examples
    --------
    Select a sample using a property measured in both catalogues, then
    reconstruct another property available only in the reference::

        reconstruction = Reconstruction(
            reference_dataset=reference,
            test_dataset=test,
            selection_variable="selection_property",
            selection_min=lower,
            selection_max=upper,
        )
        reconstruction.select_features_grid()
        reconstruction.process()
        density = reconstruction.get_distribution("reference_property", edges)

    For example, select an apparent-magnitude interval and reconstruct stellar
    mass. For redshift calibration, select a photometric-redshift interval and
    reconstruct spectroscopic redshift. All property names are user-defined.
    The selected and reconstructed properties must be numerical; no redshift
    condition is implicitly imposed. For reference PDZs supplied through
    Dataset(PDZs=..., PDZ_x=...), reconstruct redshift explicitly with::

        density = reconstruction.get_distribution(
            "redshift", reference.PDZ_x, use_pdzs=True
        )

    This also works when reference.z_gal_string is None and no point redshifts
    are available, provided selection_variable is available in both samples.
    Use use_pdzs=False to request a point-estimate histogram instead.
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
        """
        Validate parent samples and prepare feature calibration.

        Parameters and the complete reconstruction procedure are documented
        in :class:`Reconstruction`. The first internal ``PointProbability``
        uses only the full reference for feature selection. Matching is deferred
        to :meth:`process`, which supplies both selected datasets and their
        photometric errors to a separate internal ``PointProbability``.
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

        # _selection_reference copies _selection_key in a column.
        # SelectionDefinition.additional_rules will base its computations on this variable.
        # PDZs are disabled ONLY in this internal view: sample selection and
        # feature calibration use the observed selection variable. Original
        # reference CDZs remain available for get_distribution().
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
                "subsets. Analogue contributions are normalized by reference "
                "weights and aggregated using test weights; no membership "
                "probability is multiplied into the result."
            )

    def _selection_reference(self, ds):
        result = copy(ds)
        result.dataset = ds.dataset.copy(deep=False)
        result.dataset[self._selection_key] = ds.dataset[self._ref_key].to_numpy()
        result.z_gal_string = None
        result.CDZs = None
        result.PDZ_x = None
        return result

    @staticmethod
    def _resolve(ds, variable):
        """Resolve the declared redshift alias before looking up a column."""
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
        weights = ds.dataset["weights"].to_numpy(dtype=float, na_value=np.nan)
        if np.any(~np.isfinite(weights) | (weights < 0)):
            raise ValueError("Dataset weights must be finite and nonnegative.")
        if not np.isfinite(weights.sum()):
            raise ValueError("Total weight overflows; rescale input weights.")
        return weights

    def _selection_mask(self, ds, key, name):
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
        """Select colors using interval membership in the full reference.

        Parameters
        ----------
        tolerance : float, default=0.1
            Maximum fractional change allowed in purity and completeness
            relative to their initial grid-calibration values during backward
            elimination. Forwarded unchanged to :class:`~fomoz.probability.point_probability.PointProbability`.
        purity_min : float, default=0.9
            Requested minimum purity for the initial :class:`~fomoz.binary_classification.grid_classification.GridClassification`.
        grid_precision : float, default=0.15
            Resolution of the color-color grids used by :class:`~fomoz.binary_classification.grid_classification.GridClassification`.
        step : float, default=0.01
            Threshold reduction step used during grid calibration, forwarded
            through :class:`~fomoz.probability.point_probability.PointProbability`. It is not a histogram bin width.
        retry_calibration : bool, default=True
            Allow the retry procedure of
            :class:`~fomoz.probability.point_probability.PointProbability` for recognized calibration
            convergence failures. That procedure may increase ``purity_min`` and
            reports the change through a warning.

        Returns
        -------
        list of tuple(str, str)
            Retained colors, also stored in ``self.colors``. Removed colors are
            stored in ``self.discarded_colors``.

        Raises
        ------
        ValueError, TypeError
            Propagated from :class:`~fomoz.probability.point_probability.PointProbability` or :class:`~fomoz.binary_classification.grid_classification.GridClassification` for invalid
            settings or unsuccessful calibration.

        Notes
        -----
        Both positive and negative examples come from the full reference;
        target membership is ``selection_min < selection_variable <= selection_max``.
        Reference weights enter calibration metrics; test weights do not.
        This optimizes selection membership, not recovery of every feature of
        the reconstructed property distribution. If no color can be removed,
        :class:`~fomoz.probability.point_probability.PointProbability` may retain all candidate colors with a warning.

        Run before :meth:`~fomoz.reconstruction.reconstruction.Reconstruction.process`. Re-running invalidates previous matching results,
        including when the new feature calibration fails.
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
        result = copy(ds)
        result.dataset = ds.dataset.iloc[indices].copy().reset_index(drop=True)
        result.CDZs = None
        result.PDZ_x = None
        return result

    def process(
        self, min_colors_used=2, cpu_number=None, block_size=256, unmatched="drop"
    ):
        """Match selected test objects and accumulate reference contributions.

        Parameters
        ----------
        min_colors_used : int, default=2
            Minimum number of selected colors with finite values and finite
            errors for a test object. Every usable selected color constrains
            matching, not only this minimum number. Objects below the minimum
            are unsupported. Must not exceed the number of retained colors.
        cpu_number : int or None, default=None
            Number of worker threads. ``None`` uses ``max(cpu_count()-2, 1)``; explicit
            positive integers are capped at that same limit, as in
            :class:`~fomoz.probability.point_probability.PointProbability`. Set to 1 for serial execution.
        block_size : int, default=256
            Selected test objects per task. Tasks are bounded in flight; the
            full test-reference match matrix is never stored. Each active task
            holds one accumulator of length equal to the selected reference.
        unmatched : {'raise', 'drop'}, default='drop'
            ``'raise'`` fails if any positive-weight selected test object has no
            positive-weight analogue or insufficient usable colors. ``'drop'``
            explicitly reconstructs only the supported subset, with a warning
            reporting both object count and weight coverage. If no positive
            test weight is supported, both modes raise an error.

        Returns
        -------
        Reconstruction
            This instance, with ``reference_weights`` and ``diagnostics`` populated.

        Raises
        ------
        RuntimeError
            If feature selection has not run or transferred weights do not
            conserve supported test weight.
        ValueError
            If processing parameters are invalid, magnitude errors are negative,
            photometric inputs fail :class:`~fomoz.probability.point_probability.PointProbability` validation, or support is
            insufficient under the requested unmatched policy.

        Notes
        -----
        Only objects inside the interval in BOTH datasets participate. Reference
        weights are normalized separately for every test object's analogues;
        test weights scale their total contribution. No membership probability
        or further prior-weight factor is applied. A single positive-weight
        analogue is allowed and has effective count one.

        Magnitude errors use the ``'<band>_error'`` convention of :class:`~fomoz.probability.point_probability.PointProbability`.
        Nonfinite color values or errors are treated as unavailable. Zero errors
        are allowed; negative finite magnitude errors raise an exception.
        Test and reference photometric errors are always combined in quadrature
        when matching the two separate samples.

        Previous results are invalidated before matching. Unsupported objects
        have zero contributions; they are not assigned a fallback distribution.
        Changing ``self.colors`` requires re-running :meth:`~fomoz.reconstruction.reconstruction.Reconstruction.process`. Input :class:`~fomoz.data.dataset.Dataset` objects
        must remain unchanged throughout the lifetime of this reconstruction.
        """
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
        self.reference_weights = None
        self.diagnostics = None
        ref = self._selection_reference(
            self._subset(self.reference_dataset, self.reference_indices)
        )
        test = self._subset(self.test_dataset, self.test_indices)
        # Test weights ARE used below. Suppress only PointProbability's warning
        # about its own probability calculation, which we do not call.
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
            # Infinities and invalid errors must never provide a matching window.
            colors, errors = matrices["colors"], matrices["errors"]
            invalid = ~np.isfinite(colors) | ~np.isfinite(errors)
            errors[invalid] = np.nan
            colors[invalid] = np.nan
        # Reject negative band errors before squaring in the shared color builder.
        for ds in (ref, test):
            for band in {b for color in self.colors for b in color}:
                err = ds.dataset[band + "_error"].to_numpy(dtype=float)
                if np.any(np.isfinite(err) & (err < 0)):
                    raise ValueError(f"Negative magnitude errors in {band}_error.")
        rw = self._ref_weights[self.reference_indices]
        nt, nr = len(self.test_indices), len(self.reference_indices)
        counts = np.zeros(nt, dtype=int)
        neff = np.zeros(nt)
        ncolors = np.zeros(nt, dtype=int)
        supported = np.zeros(nt, dtype=bool)
        point._matching_indices(0)  # compile once before starting threads

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
                # Scale first to avoid overflow when normalizing local weights.
                alpha = rw[indices] / rw[indices].max()
                alpha /= alpha.sum()
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
                    # accumulator for every queued test block.
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
            print(
                f"Matching complete. Test-weight coverage: "
                f"{supported_weight/selected_weight:.6f}."
            )
        return self

    def get_distribution(self, variable, bin_edges, normalize=True, use_pdzs=None):
        """Reconstruct a property from weighted point estimates or reference PDZs.

        Parameters
        ----------
        variable : str
            Numerical column to reconstruct, required only in the reference.
            It can differ from ``selection_variable``, e.g. stellar mass for a
            magnitude-selected sample, or spectroscopic redshift for a sample
            selected in photometric redshift. The original redshift-column name
            supplied to :class:`~fomoz.data.dataset.Dataset` is accepted as an alias when it was renamed to ``'redshift'``.
            In PDZ mode, use "redshift" or the reference z_gal_string alias;
            numerical redshift point estimates are not required. PDZs supplied
            to Dataset describe redshift and cannot be applied to unrelated
            properties such as stellar mass. Other properties use point values.
        bin_edges : array-like of float
            One-dimensional finite, strictly increasing edges. Unequal bin
            widths are supported. NumPy conventions apply: bins are ``[left, right)``,
            except the last, which includes its right edge. These edges
            need not equal or lie inside the interval used to select the sample.
            For PDZs, edges are mapped to PDZ_x using searchsorted(side="left")
            and clipped to its endpoints, exactly as in SelectionDefinition.
            There is no interpolation; off-grid internal edges produce a warning.
            Prefer edges drawn from PDZ_x. Outside-grid CDF values are zero/one.
        normalize : bool, default=True
            If ``True``, return a probability density with unit integral over the
            supplied bin range. If ``False``, return sums of transferred test
            weights in each bin. The latter are ordinary expected counts only
            when test weights are unity; otherwise they are weighted counts.
        use_pdzs : bool or None, default=None
            None automatically uses reference PDZs for "redshift" or the
            reference Dataset's z_gal_string alias if CDZs are available.
            Otherwise use point estimates. True requires reference PDZs and a
            redshift variable; False forces point estimates and warns if
            available redshift PDZs are being bypassed. This option does not
            change the point-based parent-sample selection or the matching.
            PDZs are never inferred to belong to an arbitrary column name.

        Returns
        -------
        numpy.ndarray
            Array of length ``len(bin_edges)-1``. With ``normalize=True`` the units are
            inverse units of ``variable``. Otherwise, the units are those of the
            test weights. Edges are not returned.

        Raises
        ------
        RuntimeError
            If :meth:`~fomoz.reconstruction.reconstruction.Reconstruction.process` has not succeeded or ``self.colors`` changed afterwards.
        KeyError
            If ``variable`` is absent from the reference.
        ValueError
            If ``bin_edges`` are invalid, contributing reference objects have
            missing/nonfinite point values, reference PDZ arrays are invalid,
            PDZ mode is requested for an unrelated property or without PDZs,
            or normalization has zero mass.
        TypeError
            If variable is not a nonempty string or use_pdzs is not bool/None.

        Notes
        -----
        In point-estimate mode, missing values include sentinels declared in
        ``reference_dataset``. In PDZ mode, contributing CDF rows must instead
        be finite, nondecreasing, normalized, and aligned with the FULL reference.
        Their redshift point estimates may be missing.
        Missing properties on zero-contribution objects are allowed. Missing
        properties on contributing analogues raise an error rather than changing
        the target population silently.

        Point values or PDZ mass outside the represented range are excluded
        with a warning reporting the lost weight fraction. A normalized histogram then describes the
        distribution conditional on the supplied range. With ``unmatched='drop'``
        in :meth:`~fomoz.reconstruction.reconstruction.Reconstruction.process`, it additionally describes only the supported test subset.
        For full-range bins, unnormalized counts sum to ``supported_test_weight``.
        In PDZ mode this means covering the full PDZ_x grid. Dataset normalizes
        each PDZ on that grid; probability outside the input grid is unknown.
        Narrow bins mapping to identical CDF grid indices have zero mass.

        This method reuses the matching results and can be called for different
        reference properties or binning without repeating :meth:`~fomoz.reconstruction.reconstruction.Reconstruction.process`. No extra
        reference weight, membership probability, or survey-specific correction
        is multiplied into the histogram. No error bars are estimated.
        """
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
        # Dataset attaches PDZs to redshift, not to every numerical property.
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
        """Sum reference CDF increments with the transferred weights.

        Use the same left-search/clipping convention as SelectionDefinition
        in PointProbability. No new matching or feature selection is performed.
        Validate contributing CDF rows only; their positions refer to the full
        reference catalogue, not to the internally selected subset.
        """
        reference = self.reference_dataset
        grid = np.asarray(reference.PDZ_x, dtype=float)
        cdfs = np.asarray(reference.CDZs, dtype=float)
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
        indices = np.clip(np.searchsorted(grid, edges, side="left"), 0, grid.size - 1)
        within = (edges > grid[0]) & (edges < grid[-1])
        if np.any(within & (grid[indices] != edges)):
            warnings.warn(
                "Some bin edges do not coincide with PDZ_x. As in "
                "PointProbability/SelectionDefinition, CDFs are evaluated at "
                "the first grid point at or above each edge (no interpolation). "
                "Use edges on PDZ_x to avoid grid-dependent bin shifts.",
                stacklevel=3,
            )
        # Normalize weights before multiplication to avoid overflow. The bin
        # probabilities equal the weighted average of PointProbability values.
        weights = self.reference_weights[active]
        total_weight = weights.sum()
        edge_cdfs = rows[:, indices]
        probabilities = np.clip(np.diff(edge_cdfs, axis=1), 0.0, 1.0)
        mass = (weights / total_weight) @ probabilities
        retained = mass.sum()
        if retained > 1 + tolerance:
            raise ValueError("Reference PDZ bin probabilities exceed unit mass.")
        return mass * total_weight, max(0.0, 1.0 - retained)


