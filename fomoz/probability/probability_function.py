"""
Derive probability density functions.

This module derives probability functions based on point-estimate probabilities
obtained through the `PointProbability` class. It is ideal to derive photometric
redshifts (photo-zs) of individual galaxies, though it is general enough to estimate
distributions of other physical quantities, such as stellar masses and star formation rates.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import warnings
import numpy as np
from tqdm import tqdm
from scipy.interpolate import BSpline
from scipy.optimize import minimize
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from threading import Lock

from ..auxiliary.probability_helper import _process_multiple_points
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
        delta=0.02,
        min_variable=0.05,
        max_variable=2.5,
        verbose=True,
    ):
        r"""
        Estimate per-object probability densities from cumulative probabilities.

        Feature selection is performed independently at each sampled threshold.
        Monotone cubic B-splines are fitted to the resulting cumulative
        estimates to obtain nonnegative probability densities.

        The output of the `process` method is a probability density function
        (whose integral equals 1) for each object in `test_dataset`.
        A selection of the most informative features is recommended before the `process`
        call. In the current version of the code, the following function implements
        such a selection: `select_features_grid`.
        Regarding prior weights and reference PDZs, their contribution can be included
        in the estimates of the probability functions. See the documentation of
        `PointProbability` for more details.

        Raw cumulative estimates are retained in `cumulative_matrix`. Use
        `update_fit` to change the number of spline knots without repeating
        the point-probability calculations. Valid densities are normalized
        over the sampled interval; numpy.arange excludes max_variable.

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
        delta : float, default=0.02
            The sampling precision over the random variable's domain.
        min_variable : float, default=0.05
            Minimum of the random variable range explored to construct the
            probability functions.
        max_variable : float, default=2.5
            Maximum of the random variable range.
        verbose : bool, default=True
            Whether to display progress bars and timing information.
            This option does not suppress warnings.
        """
        self.features_were_selected = False
        self.process_was_called = False
        self.cumulative_matrix = None
        self.verbose = verbose
        if test_dataset is None:
            self.N_test_objects = len(reference_dataset.dataset)
        else:
            self.N_test_objects = len(test_dataset.dataset)

        if random_variable not in reference_dataset.dataset.columns:
            raise ValueError(f"{random_variable} is not a key of reference_dataset.")
        if (
            test_dataset is not None
            and random_variable not in test_dataset.dataset.columns
        ):
            raise ValueError(f"{random_variable} is not a key of test_dataset.")

        self.test_dataset = (  # Retain the test sample
            reference_dataset if test_dataset is None else test_dataset
        )

        if random_variable == "redshift" and min_variable <= 0.0:
            raise ValueError(
                f"Redshift PDFs can be evaluated for z>0. Got min_variable={min_variable}."
            )

        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be finite and strictly positive.")

        if (
            not np.isfinite(min_variable)
            or not np.isfinite(max_variable)
            or max_variable <= min_variable
        ):
            raise ValueError(
                "Variable bounds must be finite, with max_variable > min_variable."
            )

        self.delta = float(delta)

        # Build the dictionary
        self.variable_samples = np.arange(min_variable, max_variable, delta)

        if self.variable_samples.size < 2:
            raise ValueError("At least two variable samples are required.")

        self.dict_point = {}
        for s in self.variable_samples:
            # Define SelectionDefinition
            if random_variable == "redshift":
                sel = SelectionDefinition(z_lim_opposite=s)
            else:
                sel = SelectionDefinition(
                    additional_rules=[
                        lambda df, threshold=s, column=random_variable: df[column]
                        < threshold
                    ]
                )

            # Fill the dict
            self.dict_point[s] = PointProbability(
                sel,
                reference_dataset,
                test_dataset=test_dataset,
                verbose=False,
            )

    def select_features_grid(
        self,
        tolerance=0.1,
        purity_min=0.9,
        grid_precision=0.15,
        step=0.01,
        cpu_number=4,
    ):
        r"""
        Select features concurrently with local retries and shared purity.

        Each threshold starts from the shared purity requirement available
        when its task begins. On calibration failure, retry locally with
        `calibration_purity + 0.001` and raise the shared requirement if needed.

        Running tasks retain their own local requirement. Successful
        selections are never recomputed. Newly started tasks inherit the
        latest shared requirement.

        Parameters
        ----------
        tolerance : float, default=0.1
            Maximum relative change in purity and completeness during
            backward feature elimination.
        purity_min : float, default=0.9
            Initial minimum calibration purity.
        grid_precision : float, default=0.15
            Resolution of the color-color grids.
        step : float, default=0.01
            Threshold reduction step used during calibration.
        cpu_number : int or None, default=4
            Maximum number of concurrent threshold workers. Explicit values
            are capped at max(cpu_count() - 2, 1). None uses that limit.
            Use 1 for sequential processing.
            More workers increase the memory required for simultaneous
            calibrations and may not provide proportional speedups.

        Returns
        -------
        None
            Store selected features in the internal PointProbability objects.
            Set `features_were_selected` to True only after all tasks succeed.

        Notes
        -----
        Shared purity updates are protected by a lock. The requirement
        inherited by each task depends on task scheduling, so results
        are not guaranteed to be independent of concurrency.
        """
        if self.process_was_called:
            raise RuntimeError(
                "Internal PointProbability instances have been released. "
                "Create a new ProbabilityFunction instance."
            )

        if cpu_number is not None:
            if (
                isinstance(cpu_number, (bool, np.bool_))
                or not isinstance(cpu_number, (int, np.integer))
                or cpu_number < 1
            ):
                raise ValueError("cpu_number must be a positive integer or None.")

        if not isinstance(purity_min, float):
            raise ValueError("purity_min must be a float.")

        if not np.isfinite(purity_min) or not 0.0 < purity_min < 1.0:
            raise ValueError("purity_min must be finite and between 0 and 1.")

        samples = list(self.variable_samples)

        max_workers = max(mp.cpu_count() - 2, 1)
        n_jobs = (
            max_workers if cpu_number is None else min(int(cpu_number), max_workers)
        )
        n_jobs = max(1, min(n_jobs, len(samples)))

        self.features_were_selected = False

        shared_purity_min = purity_min
        purity_lock = Lock()
        retry_events = []

        def select_one(s):
            """Process one threshold, with independent local retries."""
            nonlocal shared_purity_min

            # Read when the task actually starts, not when submitted.
            with purity_lock:
                local_purity_min = shared_purity_min

            while True:
                try:
                    self.dict_point[s].select_features_grid(
                        tolerance=tolerance,
                        purity_min=local_purity_min,
                        grid_precision=grid_precision,
                        step=step,
                    )
                    return

                except ValueError as error:
                    # Retry only the known calibration failure.
                    if not hasattr(error, "calibration_purity"):
                        raise

                    achieved_purity = float(error.calibration_purity)
                    new_purity_min = achieved_purity + 0.001

                    if (
                        not np.isfinite(new_purity_min)
                        or new_purity_min >= 1.0
                        or new_purity_min <= local_purity_min
                    ):
                        raise ValueError(
                            f"Cannot retry feature selection at variable={s}: "
                            f"achieved purity={achieved_purity:.6f}, "
                            f"current purity_min={local_purity_min:.6f}, "
                            f"proposed purity_min={new_purity_min:.6f}. "
                            "The new threshold must increase and remain below 1."
                        ) from error

                    # Propagate the maximum to tasks not yet started.
                    with purity_lock:
                        shared_purity_min = max(
                            shared_purity_min,
                            new_purity_min,
                        )
                        retry_events.append(s)

                    local_purity_min = new_purity_min

        with tqdm(
            total=len(samples),
            desc="Selecting features for each random variable value",
            disable=not self.verbose,
        ) as progress:

            if n_jobs == 1:
                for s in samples:
                    select_one(s)
                    progress.update(1)

            else:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    futures = []

                    try:
                        for s in samples:
                            futures.append(executor.submit(select_one, s))

                        for future in as_completed(futures):
                            future.result()
                            progress.update(1)

                    except BaseException:
                        # Cancel queued tasks; running tasks finish
                        # before the executor exits.
                        for future in futures:
                            future.cancel()
                        raise

        self.features_were_selected = True

        if retry_events:
            warnings.warn(
                f"Calibration required {len(retry_events)} retries across "
                f"{len(set(retry_events))} sampled thresholds. "
                f"The shared purity requirement increased from "
                f"{purity_min:.6f} to {shared_purity_min:.6f}. "
                "All feature selections completed successfully.",
                stacklevel=2,
            )

    def process(
        self,
        min_colors_used=2,
        cpu_number=None,
        block_size=256,
        n_internal_knots=10,
    ):
        r"""
        Compute and retain cumulative estimates, then fit probability densities.

        Call `_process_multiple_points` at each sampled threshold, preserving
        threshold-specific feature selections. Save the raw estimates in
        `cumulative_matrix`, then call `update_fit` for monotone cubic spline
        fitting and analytic differentiation.

        Parameters
        ----------
        min_colors_used : int, default=2
            Minimum number of usable test colors (see `PointProbability.process`
            for details).
        cpu_number : int or None, optional
            Number of worker threads. If None, use the number of logical CPUs
            minus two, with a minimum of one. Explicit values are capped at that
            limit. Spline fits are performed sequentially.
        block_size : int, default=256
            Number of test objects per task.
        n_internal_knots : int, default=10
            Number of equally spaced interior knots for the initial fit.
            These divide the sampled domain into n_internal_knots + 1
            intervals. See `update_fit` for details and fitting constraints.

        Returns
        -------
        None
            Store raw cumulative estimates in `cumulative_matrix`, with
            shape (n_samples, n_test). Rows are thresholds and columns are
            objects. Missing estimates remain NaN. Also set the density
            outputs and diagnostics documented in `update_fit`.

        Raises
        ------
        RuntimeError
            If processing has already started on this instance.
        ValueError
            If the knot count or sampling grid is invalid.

        Notes
        -----
        Internal PointProbability instances are released after cumulative estimation.
        This method can only be started once. Subsequent calls to `update_fit`
        reuse the saved cumulative estimates without repeating matching or
        feature selection. The raw matrix requires approximately
        8 * n_samples * n_test bytes and is retained for these later fits.
        If cumulative estimation fails partway through, create a new instance.
        Once all cumulative estimates are saved, fitting can be retried using
        `update_fit`, even if the initial fitting stage raises an exception.
        """
        if self.process_was_called:
            raise RuntimeError(
                "Processing has already been started on this instance. "
                "Use update_fit to refit saved cumulative estimates, or "
                "create a new instance to recompute them."
            )

        # Validate before consuming or releasing any PointProbability instance.
        self._validate_fit_grid(n_internal_knots)

        if not self.features_were_selected:
            warnings.warn(
                "No feature selection was performed. Using all available features."
            )

        self.process_was_called = True

        # Rows are thresholds; columns are test objects
        cumulative_matrix = _process_multiple_points(
            [self.dict_point[s] for s in self.variable_samples],
            min_colors_used=min_colors_used,
            cpu_number=cpu_number,
            block_size=block_size,
            verbose=self.verbose,
        )

        for s in self.variable_samples:  # release the PointProbability objects
            self.dict_point[s] = None

        self.cumulative_matrix = cumulative_matrix
        self.update_fit(n_internal_knots)

    def _validate_fit_grid(self, n_internal_knots):
        """Validate the knot count and grid without modifying stored results."""
        if (
            isinstance(n_internal_knots, (bool, np.bool_))
            or not isinstance(n_internal_knots, (int, np.integer))
            or n_internal_knots < 0
        ):
            raise ValueError("n_internal_knots must be a non-negative integer.")

        x = np.asarray(self.variable_samples, dtype=float)
        n_coefficients = int(n_internal_knots) + 4

        if (
            x.ndim != 1
            or x.size < n_coefficients
            or not np.all(np.isfinite(x))
            or np.any(np.diff(x) <= 0)
        ):
            raise ValueError(
                "variable_samples must be finite, strictly increasing, "
                "and contain at least n_internal_knots + 4 points."
            )

        return x, n_coefficients

    def update_fit(self, n_internal_knots):
        r"""
        Refit saved cumulative estimates with monotone cubic B-splines.

        This method is called by `process` for the initial fit and can be
        called repeatedly afterwards. It reuses the unchanged raw estimates
        in `cumulative_matrix`; no PointProbability instance is required.
        Each call starts a new fit, independently of previous fitted results.

        Parameters
        ----------
        n_internal_knots : int
            Number of equally spaced knots strictly inside the fitting
            domain. The endpoints are not counted. The knots divide the
            domain into ``n_internal_knots + 1`` intervals, each described
            by a cubic polynomial. Adjacent pieces are joined with
            continuous first and second derivatives and fitted jointly
            under monotonicity constraints.

            For example, over [0, 2.5], five internal knots create six
            intervals of width approximately 0.417. This differs from
            `delta`, which controls the spacing of the thresholds at
            which point probabilities are estimated.

            More knots allow greater flexibility and can resolve finer
            structure, but may introduce artificial peaks in the derived
            PDF. Fewer knots can broaden, merge, or shift real peaks.
            Knot spacing is not a strict minimum peak width.

            The cubic spline has ``n_internal_knots + 4`` coefficients.
            Each object requires at least this many finite cumulative
            estimates, including estimates at both domain endpoints.

        Returns
        -------
        None
            Replace the following attributes after the fitting loop completes:

            probability_functions : numpy.ndarray
                Shape (n_test, n_samples). Nonnegative density samples with
                unit trapezoidal integral over variable_samples. Invalid
                rows contain NaN, even if an earlier fit succeeded.
            fit_success : numpy.ndarray of bool
                Shape (n_test,). Whether fitting and normalization succeeded.
            interval_probability : numpy.ndarray
                Shape (n_test,). Fitted cumulative probability difference
                between the domain endpoints, before density normalization.
                NaN for invalid fits.
            n_internal_knots : int
                Knot count used for the newly stored results.

        Raises
        ------
        RuntimeError
            If complete cumulative estimates have not been saved by process.
        ValueError
            If the knot count, sampling grid, or cumulative matrix shape is
            invalid. Previously stored fits are preserved in this case.

        Notes
        -----
        The fitted cumulative has the form

        .. math::
            F(x) = \sum_m c_m B_m(x),
            \qquad 0 \leq c_0 \leq \cdots \leq c_{M-1} \leq 1.

        These coefficient constraints are sufficient to guarantee
        monotonicity throughout the sampled interval, not only at the
        sampling points.

        The spline coefficients are estimated independently for each object
        by minimizing the equally weighted least-squares objective

        .. math::
            \frac{1}{2}\sum_{k \in V}
            \left[F(x_k) - \widehat{F}(x_k)\right]^2,

        where V contains the sampling points with finite cumulative estimates
        and \widehat{F}(x_k) denotes the probabilities returned by
        PointProbability.

        Optimization uses `scipy.optimize.minimize` with the SLSQP method,
        an analytic objective gradient, coefficient bounds [0, 1], and
        linear inequality constraints enforcing nondecreasing coefficients.
        Knot positions and spline degree remain fixed during optimization.
        Fits that fail to converge are rejected and produce NaN densities.

        Finite cumulative estimates receive equal weights in the fit.
        Missing interior estimates are omitted. No covariance model or
        uncertainty propagation is applied.

        Endpoint probabilities are fitted rather than fixed to zero and
        one. The output density is normalized over the sampled interval,
        so it represents the distribution conditional on that interval.
        Trapezoidal normalization of the sampled derivative introduces
        a grid-dependent numerical approximation.

        Objects with insufficient valid estimates, missing endpoint
        estimates, failed optimization, or fitted interval probability
        below the numerical tolerance receive NaN densities.

        Fits are performed sequentially. This method leaves the raw cumulative
        matrix unchanged. Do not change variable_samples after processing.
        """
        if self.cumulative_matrix is None:
            raise RuntimeError(
                "No complete cumulative estimates are available. "
                "Run process successfully through cumulative estimation first."
            )

        x, n_coefficients = self._validate_fit_grid(n_internal_knots)
        cumulative_matrix = self.cumulative_matrix
        if cumulative_matrix.shape != (x.size, self.N_test_objects):
            raise ValueError("cumulative_matrix must have shape (n_samples, n_test).")

        # Construct clamped cubic B-spline bases on the sampled domain.
        internal_knots = np.linspace(x[0], x[-1], int(n_internal_knots) + 2)[1:-1]

        knots = np.concatenate(
            (
                np.repeat(x[0], 4),
                internal_knots,
                np.repeat(x[-1], 4),
            )
        )

        basis = BSpline(
            knots,
            np.eye(n_coefficients),
            k=3,
            extrapolate=False,
        )

        # Shared matrices: evaluate the cumulative and its analytic derivative.
        design = basis(x)
        derivative_design = basis.derivative()(x)

        # Enforce c[m + 1] - c[m] >= 0.
        difference_matrix = np.diff(np.eye(n_coefficients), axis=0)
        constraints = {
            "type": "ineq",
            "fun": lambda coefficients: difference_matrix @ coefficients,
            "jac": lambda coefficients: difference_matrix,
        }
        bounds = [(0.0, 1.0)] * n_coefficients

        probability_functions = np.full((self.N_test_objects, x.size), np.nan)
        fit_success = np.zeros(self.N_test_objects, dtype=bool)
        interval_probability = np.full(self.N_test_objects, np.nan)

        for i in tqdm(
            range(self.N_test_objects),
            desc="Fitting monotone cumulative distributions",
            disable=not self.verbose,
        ):
            y = cumulative_matrix[:, i]
            finite = np.isfinite(y)

            # Require enough observations and avoid extrapolating endpoints.
            if (
                np.count_nonzero(finite) < n_coefficients
                or not finite[0]
                or not finite[-1]
            ):
                continue

            observed = y[finite]
            if np.any(observed < -1e-8) or np.any(observed > 1.0 + 1e-8):
                continue

            observed = np.clip(observed, 0.0, 1.0)
            matrix = design[finite]

            # Feasible initial coefficients, including for nonmonotone data.
            initial = np.linspace(observed.min(), observed.max(), n_coefficients)

            def objective(coefficients):
                residuals = matrix @ coefficients - observed
                loss = 0.5 * np.sum(residuals * residuals)
                gradient = matrix.T @ residuals
                return loss, gradient

            result = minimize(
                objective,
                initial,
                method="SLSQP",
                jac=True,
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-10, "maxiter": 200},
            )

            if not result.success or not np.all(np.isfinite(result.x)):
                continue

            coefficients = result.x

            # Reject constraint violations larger than numerical tolerance.
            if (
                np.any(coefficients < -1e-8)
                or np.any(coefficients > 1.0 + 1e-8)
                or np.any(np.diff(coefficients) < -1e-8)
            ):
                continue

            # Remove only residual numerical violations in fitted coefficients.
            coefficients = np.maximum.accumulate(np.clip(coefficients, 0.0, 1.0))

            # Clamped endpoint values equal the first and last coefficients.
            interval_mass = coefficients[-1] - coefficients[0]
            if interval_mass <= 1e-10:
                continue

            density = derivative_design @ coefficients

            # The model derivative is nonnegative; clip numerical roundoff.
            density = np.maximum(density, 0.0)
            normalization = np.trapz(density, x)

            if not np.isfinite(normalization) or normalization <= 0:
                continue

            probability_functions[i] = density / normalization
            interval_probability[i] = interval_mass
            fit_success[i] = True

        self.probability_functions = probability_functions
        self.fit_success = fit_success
        self.interval_probability = interval_probability
        self.n_internal_knots = int(n_internal_knots)

        n_invalid = np.count_nonzero(~fit_success)
        if n_invalid:
            warnings.warn(
                f"{n_invalid} object(s) had insufficient valid cumulative "
                "estimates, an unsuccessful fit, or negligible fitted "
                "interval probability; their densities were set to NaN."
            )

    def get_results_dataframe(self):
        r"""
        Return the test DataFrame with scalar summaries of the fitted PDFs.

        Preserve the original columns, row order, and index, and append
        scalar results with the ``fomoz_`` prefix. Summaries are computed
        from the current `probability_functions`, including any changes
        made by the latest successful `update_fit` call.

        Full PDFs are not stored in the returned DataFrame.

        Returns
        -------
        pandas.DataFrame
            Copy of the test DataFrame with the following added columns:

            fomoz_mode
                Sampled variable value at the maximum PDF density.
                Its resolution is limited by `variable_samples`. If several
                samples share the maximum, use the first, at the lowest
                variable value.
            fomoz_mean
                Mean of the distribution.
            fomoz_median
                Median of the distribution.
            fomoz_std
                Standard deviation of the distribution. This describes
                its width, not the numerical error on the inferred mean.
            fomoz_p16, fomoz_p84
                Quantiles defining the central 68% probability interval.
                This is not necessarily a highest-density interval.
            fomoz_fit_success
                Whether fitting and PDF normalization succeeded.
            fomoz_interval_probability
                Fitted cumulative probability difference between the
                sampled endpoints, before PDF normalization.

            Distribution summaries are NaN for unsuccessful fits.

        Raises
        ------
        RuntimeError
            If fitted probability densities are not available.
        ValueError
            If the sampling grid, result shapes, or valid PDF rows are
            inconsistent, or an output column already exists.

        Notes
        -----
        Summaries describe the distribution conditional on the sampled
        interval: the fitted PDFs are normalized within that interval.
        They do not describe unmodelled tails outside the interval.

        Means and variances use trapezoidal quadrature on the sampled
        PDF. Quantiles are obtained by integrating the piecewise-linear
        interpolant of the sampled density and inverting its cumulative.
        All summaries therefore depend on the sampling resolution.

        For multimodal distributions, scalar summaries can hide important
        structure; inspect the full PDFs when needed.

        Access the full distributions through:

        - ``self.variable_samples``: common variable grid, shape (n_samples,).
        - ``self.probability_functions``: PDF densities, shape
          (n_test, n_samples). These are densities, not per-bin probabilities.

        Row i of the PDF matrix corresponds to row i of the returned
        DataFrame, accessed with ``iloc[i]``. Use positional indexing:
        DataFrame index labels need not be unique or consecutive.

        Do not modify or reorder the test DataFrame between `process`
        and this call. Changes preserving its row count cannot be
        detected by this method.

        This method modifies neither the test dataset nor the fitted PDFs.

        Examples
        --------
        >>> pdf.process()
        >>> results = pdf.get_results_dataframe()

        Retrieve one galaxy and its full PDF by position:

        >>> i = 0
        >>> galaxy = results.iloc[i]
        >>> x = pdf.variable_samples
        >>> density = pdf.probability_functions[i]

        Retrieve PDFs for a filtered subset without losing row alignment:

        >>> keep = results["fomoz_fit_success"].to_numpy()
        >>> selected_galaxies = results.iloc[np.flatnonzero(keep)]
        >>> selected_pdfs = pdf.probability_functions[keep]

        Recompute summaries after changing the spline fit:

        >>> pdf.update_fit(n_internal_knots=15)
        >>> results = pdf.get_results_dataframe()
        """
        required = (
            "probability_functions",
            "fit_success",
            "interval_probability",
        )
        if any(getattr(self, name, None) is None for name in required):
            raise RuntimeError(
                "Fitted probability densities are not available. "
                "Complete process() or update_fit() before exporting results."
            )

        test_frame = self.test_dataset.dataset
        n_objects = len(test_frame)

        x = np.asarray(self.variable_samples, dtype=float)
        densities = np.asarray(self.probability_functions)
        fit_success = np.asarray(self.fit_success, dtype=bool)
        interval_probability = np.asarray(self.interval_probability)

        if (
            x.ndim != 1
            or x.size < 2
            or not np.all(np.isfinite(x))
            or np.any(np.diff(x) <= 0)
        ):
            raise ValueError(
                "variable_samples must be finite, strictly increasing, "
                "and contain at least two points."
            )

        if densities.shape != (n_objects, x.size):
            raise ValueError(
                "probability_functions must have shape "
                f"({n_objects}, {x.size}); got {densities.shape}."
            )

        if fit_success.shape != (n_objects,) or interval_probability.shape != (
            n_objects,
        ):
            raise ValueError("Fit diagnostics must have one entry per test object.")

        summary_names = (
            "mode",
            "mean",
            "median",
            "std",
            "p16",
            "p84",
        )
        output_columns = [f"fomoz_{name}" for name in summary_names] + [
            "fomoz_fit_success",
            "fomoz_interval_probability",
        ]

        collisions = [
            column for column in output_columns if column in test_frame.columns
        ]
        if collisions:
            raise ValueError(
                "The test DataFrame already contains output columns: "
                f"{collisions}. Rename or remove them before exporting."
            )

        summaries = {name: np.full(n_objects, np.nan) for name in summary_names}
        dx = np.diff(x)

        for i in np.flatnonzero(fit_success):
            density = np.asarray(densities[i], dtype=float)

            if not np.all(np.isfinite(density)) or np.any(density < 0):
                raise ValueError(
                    f"PDF row {i} is marked successful but contains "
                    "nonfinite or negative densities."
                )

            segment_mass = 0.5 * (density[:-1] + density[1:]) * dx
            normalization = segment_mass.sum()

            if not np.isfinite(normalization) or normalization <= 0:
                raise ValueError(f"PDF row {i} has an invalid normalization.")

            # Normalize locally without changing the stored PDF.
            p = density / normalization

            mean = np.trapz(x * p, x)
            variance = np.trapz((x - mean) ** 2 * p, x)

            summaries["mode"][i] = x[np.argmax(p)]
            summaries["mean"][i] = mean
            summaries["std"][i] = np.sqrt(max(variance, 0.0))

            cdf = np.concatenate(
                (
                    [0.0],
                    np.cumsum(segment_mass / normalization),
                )
            )

            # Remove residual floating-point endpoint errors.
            cdf = np.clip(cdf, 0.0, 1.0)
            cdf[-1] = 1.0

            for name, quantile in (
                ("p16", 0.16),
                ("median", 0.50),
                ("p84", 0.84),
            ):
                upper = int(np.searchsorted(cdf, quantile, side="left"))
                upper = min(max(upper, 1), x.size - 1)
                lower = upper - 1

                # Fraction of this segment's probability to accumulate.
                fraction = (quantile - cdf[lower]) / (cdf[upper] - cdf[lower])
                fraction = float(np.clip(fraction, 0.0, 1.0))

                # Invert the integral of a linear density on the segment.
                # Scale endpoint densities for numerical stability.
                scale = max(p[lower], p[upper])
                a = p[lower] / scale
                b = p[upper] / scale
                target_area = 0.5 * fraction * (a + b)

                discriminant = max(
                    a * a + 2.0 * (b - a) * target_area,
                    0.0,
                )
                denominator = a + np.sqrt(discriminant)

                position = 2.0 * target_area / denominator if denominator > 0 else 0.0
                position = float(np.clip(position, 0.0, 1.0))

                summaries[name][i] = x[lower] + position * dx[lower]

        output = test_frame.copy(deep=True)

        for name, values in summaries.items():
            output[f"fomoz_{name}"] = values

        output["fomoz_fit_success"] = fit_success.copy()
        output["fomoz_interval_probability"] = interval_probability.copy()

        return output
