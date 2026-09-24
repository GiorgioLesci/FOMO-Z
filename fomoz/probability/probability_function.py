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

                    warnings.warn(
                        f"Calibration failed at variable={s}. "
                        f"Updating its local purity_min from "
                        f"{local_purity_min:.6f} to {new_purity_min:.6f} "
                        "and retrying. Newly started tasks inherit "
                        "the highest requirement reached so far."
                    )

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
