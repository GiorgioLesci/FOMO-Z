# FOMO-Z

[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://giorgiolesci.github.io/FOMO-Z/)

A Python package for optimized galaxy classification, photometric redshift estimation, and reconstruction of population distributions.

## Description

`FOMO-Z` (Flexible Optimizer of Multi-dimensional Operators and photo-Zs) uses reference samples to relate galaxy photometry to physical properties. It provides tools for calibrating binary color selections, estimating individual membership probabilities and property distributions, and reconstructing population distributions for selected test samples.

The package supports reference weights and reference redshift probability distributions (PDZs). Its probabilistic matching methods account for photometric uncertainties and support color feature selection.

## Binary selection: `binary_classification`

The `binary_classification` subpackage calibrates color-based selections that maximize completeness given a prescribed minimum purity. These selections can target galaxies satisfying redshift cuts or criteria involving other reference properties.

### `LinearClassification`

`LinearClassification` calibrates combinations of linear color cuts. It provides interpretable selection boundaries and is particularly useful for selecting background galaxies for weak lensing analyses.

![Optimization of linear color cuts](https://github.com/user-attachments/assets/255bd07f-b66d-40ac-a539-f1429e2cd455)

The animation shows an example of the optimization procedure for a lower redshift threshold of 0.3.

### `GridClassification`

`GridClassification` calibrates selection masks on discretized color–color grids. It allows more flexible boundaries than linear cuts and can be used for selections involving properties such as stellar mass or galaxy type.

### `Classifier`

`Classifier` applies calibrated color selections to a galaxy sample and returns a Boolean mask. Calibration parameters can be passed directly from the calibration classes or saved to file and loaded later.

Both calibration classes also provide `get_distribution()` to compute weighted distributions of properties in the color-selected reference sample. For redshift distributions, reference PDZs are used whenever available; otherwise, point estimates are used.

## Probabilistic inference: `probability`

The `probability` subpackage uses photometrically compatible reference objects to infer properties of test galaxies. Matching accounts for photometric uncertainties, and feature selection identifies informative colors for the requested task.

### `PointProbability`

`PointProbability` estimates the probability that an individual test galaxy satisfies a specified selection criterion.

For each test object, it identifies reference objects with compatible colors and computes their weighted membership fraction. Reference prior weights and, for redshift criteria, reference PDZs can contribute to this estimate.

Applications include estimating the probability that a galaxy lies beyond a redshift threshold or belongs to a specified galaxy population. In the plot below, we show the quiescent/star-forming classification obtained with `PointProbability`, using COSMOS15 as the reference sample and COSMOS-Web as the test sample.

![quiescence](https://github.com/user-attachments/assets/b6cc73bc-328e-4ddb-bba8-c75d088cc4cb)

### `ProbabilityFunction`

`ProbabilityFunction` estimates the probability density function of a chosen property for individual test galaxies. It evaluates cumulative probabilities across property thresholds and fits a monotonic cumulative distribution, whose derivative provides the probability density.

For redshift inference, it produces individual photometric redshift distributions.

### `Reconstruction`

`Reconstruction` infers population distributions for a selected test sample using property measurements available in a distinct reference sample.

The `Reconstruction` class can be used as an alternative to self-organizing maps.

The user supplies the full reference and test samples, together with an observed variable and an interval defining the target selection. Feature selection uses the full reference sample. Matching is then performed between the selected subsets. Reference PDZs can be integrated when reconstructing redshift distributions.

Unlike the `get_distribution()` method of `LinearClassification` and `GridClassification`, `Reconstruction` transfers information from a reference sample to a separate test population.

## Calibration performance

The figure below shows execution time (top) and peak memory usage (bottom) as a function of the number of photometric bands. COSMOS20 is used as the reference sample and, for the probability routines, also as the test sample.

![FOMO-Z performance](https://github.com/user-attachments/assets/fec19a2f-1631-48f7-966b-0e102459ead9)

The panels compare `LinearClassification` and `GridClassification` (left), `PointProbability` and `Reconstruction` (middle), and the feature-selection and processing stages of `ProbabilityFunction` (right). CPU configurations are indicated in the legends. Memory is measured using PSS, which accounts proportionally for shared memory. Default input parameters are used where possible. `ProbabilityFunction` repeats feature selection at each sampled threshold, making its computational cost relatively substantial compared to `PointProbability` and `Reconstruction`.

The `LinearClassification` and `GridClassification` calibrations are computed only once, and are then applied to any test dataset using the \texttt{Classifier} class. Since this class relies on vectorized array operations over precomputed color conditions and grids, its execution time is negligible compared to that of the calibration step itself.

## Installation

### Get the source code

```bash
git clone https://github.com/GiorgioLesci/FOMO-Z.git
cd FOMO-Z
```

### Install in a conda environment

```bash
conda create -n fomoz python=3.9 "numpy<2" scipy astropy pyyaml -c conda-forge
conda activate fomoz
python -m pip install .
```

To reactivate the environment in a later session:

```bash
conda activate fomoz
```

### Install in an existing environment

From the repository directory:

```bash
python -m pip install .
```

### Development installation

For an editable installation with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Documentation

Full documentation, including the API reference, is available at **[giorgiolesci.github.io/FOMO-Z](https://giorgiolesci.github.io/FOMO-Z/)**.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Bug reports, questions, and feature suggestions are welcome through the [issue tracker](https://github.com/GiorgioLesci/FOMO-Z/issues). For bug reports, please include the `FOMO-Z` version, your Python, numpy, and scipy versions, a minimal reproducible example, and the full traceback.

`FOMO-Z` does not currently accept pull requests from non-collaborators. If you have a fix or an improvement in mind, please describe it in an issue instead, and I will consider implementing it. You are of course free to fork the repository under the terms of the [MIT License](LICENSE).

## Citation

If you use `FOMO-Z` in your research, please cite the accompanying paper (Lesci 2027). Citation metadata are also available in [CITATION.cff](CITATION.cff) and via the "Cite this repository" button on GitHub.
