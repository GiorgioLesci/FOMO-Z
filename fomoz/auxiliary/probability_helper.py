"""
Auxiliary functions for the probability subpackage.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import multiprocessing as mp
import numpy as np

from numba import njit
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext


@njit(cache=True, fastmath=False, nogil=True)
def _matching_indices_numba(
    reference_colors,
    reference_errors,
    test_colors,
    test_errors,
    combine_errors,
):
    """
    Return matching reference indices using a compiled sequential scan.

    Parameters
    ----------
    reference_colors, reference_errors : numpy.ndarray
        Reference matrices with shape (n_colors, n_reference).
    test_colors, test_errors : numpy.ndarray
        Color values and errors of one test object, with shape (n_colors,).
    combine_errors : bool
        Whether to combine test and reference errors in quadrature.
        Otherwise, use only the test errors.

    Returns
    -------
    numpy.ndarray
        Matching reference indices in ascending order.

    Notes
    -----
    Test colors with a NaN value or error impose no constraint.
    A reference is rejected as soon as one usable color fails.
    No reference weights or target memberships are used here.
    """
    n_reference = reference_colors.shape[1]

    # Determine usable colors once for this test object
    usable = np.flatnonzero(~np.isnan(test_colors) & ~np.isnan(test_errors))

    # Allocate enough space for the maximum possible number of matches
    indices = np.empty(n_reference, dtype=np.int64)
    n_matches = 0

    for j in range(n_reference):
        accepted = True

        for position in range(usable.size):
            k = usable[position]

            difference = abs(reference_colors[k, j] - test_colors[k])

            if combine_errors:
                tolerance = np.sqrt(test_errors[k] ** 2 + reference_errors[k, j] ** 2)
            else:
                tolerance = test_errors[k]

            # Keep this form:
            # "difference > tolerance" would not reject NaNs
            if not (difference <= tolerance):
                accepted = False
                break

        if accepted:
            indices[n_matches] = j
            n_matches += 1

    # Return only the populated portion of the buffer.
    return indices[:n_matches]


@njit(cache=True, fastmath=False, nogil=True)
def _matching_indices_cached_numba(
    reference_colors,
    reference_errors,
    test_colors,
    test_errors,
    combine_errors,
    selected_indices,
    cache,
):
    """
    Match one feature group, reusing per-color comparisons.

    cache[k, j]:
         0: not evaluated
         1: compatible
        -1: incompatible

    Reset the cache before processing a different test object.
    Missing test colors impose no constraint.
    """
    n_reference = reference_colors.shape[1]
    indices = np.empty(n_reference, dtype=np.int64)
    n_matches = 0

    for j in range(n_reference):
        accepted = True

        for position in range(selected_indices.size):
            k = selected_indices[position]

            if np.isnan(test_colors[k]) or np.isnan(test_errors[k]):
                continue

            state = cache[k, j]

            if state == 0:
                difference = abs(reference_colors[k, j] - test_colors[k])

                if combine_errors:
                    tolerance = np.sqrt(
                        test_errors[k] ** 2 + reference_errors[k, j] ** 2
                    )
                else:
                    tolerance = test_errors[k]

                # Preserve the original NaN rejection semantics
                if difference <= tolerance:
                    state = 1
                else:
                    state = -1

                cache[k, j] = state

            if state == -1:
                accepted = False
                break

        if accepted:
            indices[n_matches] = j
            n_matches += 1

    return indices[:n_matches]


def _process_multiple_points(
    points,
    min_colors_used=2,
    cpu_number=None,
    block_size=256,
    verbose=True,
):
    """Compute multiple targets using shared, lazy color matching."""
    points = list(points)
    if not points:
        raise ValueError("At least one PointProbability is required.")

    # Same parallelization settings as PointProbability.process()
    if cpu_number is not None:
        if (
            isinstance(cpu_number, (bool, np.bool_))
            or not isinstance(cpu_number, (int, np.integer))
            or cpu_number < 1
        ):
            raise ValueError("cpu_number must be a positive integer or None.")

    if (
        isinstance(block_size, (bool, np.bool_))
        or not isinstance(block_size, (int, np.integer))
        or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer.")

    max_workers = max(mp.cpu_count() - 2, 1)
    n_jobs = max_workers if cpu_number is None else min(int(cpu_number), max_workers)

    first = points[0]

    # Sharing comparisons requires identical photometry and error conventions
    for point in points:
        if (
            point.reference_dataset is not first.reference_dataset
            or point.test_dataset is not first.test_dataset
            or point._combine_reference_errors != first._combine_reference_errors
        ):
            raise ValueError("All points must share datasets and matching conventions.")

    # Union of selected colors, preserving first-appearance order
    all_colors = []
    color_positions = {}
    groups = {}

    for row, point in enumerate(points):
        colors = (
            point.reference_dataset.colors if point.colors is None else point.colors
        )
        colors = [tuple(color) for color in colors]

        for color in colors:
            if color not in color_positions:
                color_positions[color] = len(all_colors)
                all_colors.append(color)

        # Preserve the first group's color order for early rejection
        key = frozenset(colors)
        if key not in groups:
            selected = np.asarray(
                [color_positions[color] for color in colors],
                dtype=np.int64,
            )
            groups[key] = (selected, [])

        groups[key][1].append(row)

    groups = list(groups.values())

    def build_arrays(dataset):
        frame = dataset.dataset
        colors = np.empty(
            (len(all_colors), len(frame)),
            dtype=np.float64,
            order="F",
        )
        errors = np.empty_like(colors, order="F")

        for k, (band1, band2) in enumerate(all_colors):
            colors[k] = frame[band1].values - frame[band2].values
            errors[k] = np.sqrt(
                frame[band1 + "_error"].values ** 2
                + frame[band2 + "_error"].values ** 2
            )

        return colors, errors

    reference_colors, reference_errors = build_arrays(first.reference_dataset)

    if first.test_dataset is first.reference_dataset:
        test_colors, test_errors = reference_colors, reference_errors
    else:
        test_colors, test_errors = build_arrays(first.test_dataset)

    reference_weights = np.asarray(first.reference_dataset.dataset["weights"].values)

    # These masks already include reference weights
    targets = [np.asarray(point.target_mask) for point in points]

    n_test = test_colors.shape[1]
    n_reference = reference_colors.shape[1]
    combine_errors = first._combine_reference_errors

    cumulative = np.full((len(points), n_test), np.nan)

    # Compile/load the specialization before starting worker threads
    if n_test:
        warmup_cache = np.zeros(
            (len(all_colors), n_reference),
            dtype=np.int8,
            order="F",
        )
        _matching_indices_cached_numba(
            reference_colors,
            reference_errors,
            test_colors[:, 0],
            test_errors[:, 0],
            combine_errors,
            groups[0][0],
            warmup_cache,
        )
        del warmup_cache

    def process_block(start):
        stop = min(start + block_size, n_test)

        # Private to this task; reused for successive test objects.
        cache = np.empty(
            (len(all_colors), n_reference),
            dtype=np.int8,
            order="F",
        )

        for i in range(start, stop):
            cache.fill(0)

            colors_i = test_colors[:, i]
            errors_i = test_errors[:, i]
            usable = ~np.isnan(colors_i) & ~np.isnan(errors_i)

            for selected, rows in groups:
                if np.count_nonzero(usable[selected]) < min_colors_used:
                    continue

                indices = _matching_indices_cached_numba(
                    reference_colors,
                    reference_errors,
                    colors_i,
                    errors_i,
                    combine_errors,
                    selected,
                    cache,
                )

                if indices.size == 0:
                    continue

                weight_sum = reference_weights[indices].sum()

                for row in rows:
                    cumulative[row, i] = targets[row][indices].sum() / weight_sum

        return stop - start

    block_starts = range(0, n_test, block_size)
    pool_context = (
        ThreadPoolExecutor(max_workers=int(n_jobs)) if n_jobs > 1 else nullcontext()
    )

    with pool_context as executor:
        if executor is None:
            results = map(process_block, block_starts)
        else:
            results = executor.map(process_block, block_starts)

        with tqdm(
            total=n_test,
            desc="Estimating cumulative probabilities",
            disable=not verbose,
        ) as progress:
            for completed in results:
                progress.update(completed)

    return cumulative
