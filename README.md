# FOMO-Z

A Python package for optimized galaxy classification, photometric redshift estimation, and reconstruction of population distributions.

## Description

`FOMO-Z` (Flexible Optimizer of Multi-dimensional Operators and photo-Zs) uses reference samples to relate galaxy photometry to physical properties. It provides tools for calibrating binary color selections, estimating individual membership probabilities and property distributions, and reconstructing population distributions for selected test samples.

The package supports reference weights, missing photometric measurements, and reference redshift probability distributions (PDZs). Its probabilistic matching methods account for photometric uncertainties and support color feature selection.

## Binary selection: `binary_classification`

The `binary_classification` subpackage calibrates color-based selections that maximize completeness subject to a prescribed minimum purity. These selections can target galaxies satisfying redshift cuts or criteria involving other reference properties.

### `LinearClassification`

`LinearClassification` calibrates combinations of linear color cuts. It provides interpretable selection boundaries and is particularly useful for selecting background galaxies for weak lensing analyses.

![Optimization of linear color cuts](https://github.com/user-attachments/assets/255bd07f-b66d-40ac-a539-f1429e2cd455)

The animation shows an example of the optimization procedure for a lower redshift threshold of 0.3.

### `GridClassification`

`GridClassification` calibrates selection masks on discretized color–color grids. It allows more flexible boundaries than linear cuts and can be used for selections involving properties such as stellar mass or galaxy type.

### `Classifier`

`Classifier` applies calibrated color selections to a galaxy sample and returns a Boolean mask. Calibration parameters can be passed directly from the calibration classes or saved to file and loaded later.

Both calibration classes also provide `get_distribution()` to compute weighted distributions of properties in the color-selected reference sample. For redshift distributions, reference PDZs are used whenever available; otherwise, point estimates are used.

### Calibration performance

The figure below shows execution time (top) and peak RAM usage (bottom) as a function of the number of photometric bands, using COSMOS20 as the reference sample.

![Binary selection calibration performance](https://github.com/user-attachments/assets/a5142986-f632-409a-bc73-a47b51510014)

Blue and red dots show `LinearClassification` results using 1 and 12 CPUs, respectively. Green crosses show `GridClassification`, whose calibration is not parallelized in this benchmark.

Once calibrated for a given reference sample and selection configuration, the resulting color selections can be applied to compatible photometric datasets through `Classifier` without repeating the calibration.

## Probabilistic inference: `probability`

The `probability` subpackage uses photometrically compatible reference objects to infer properties of test galaxies. Matching accounts for photometric uncertainties, and feature selection identifies informative colors for the requested task.

### `PointProbability`

`PointProbability` estimates the probability that an individual test galaxy satisfies a specified selection criterion.

For each test object, it identifies reference objects with compatible colors and computes their weighted membership fraction. Reference prior weights and, for redshift criteria, reference PDZs can contribute to this estimate.

Applications include estimating the probability that a galaxy lies beyond a redshift threshold or belongs to a specified galaxy population.

### `ProbabilityFunction`

`ProbabilityFunction` estimates the probability density function of a chosen property for individual test galaxies. It evaluates cumulative probabilities across property thresholds and fits a monotonic cumulative distribution, whose derivative provides the probability density.

For redshift inference, it produces individual photometric redshift distributions.

### `Reconstruction`

`Reconstruction` infers population distributions for a selected test sample using property measurements available in a distinct reference sample.

The user supplies the full reference and test samples, together with an observed variable and an interval defining the target selection. Feature selection uses the full reference sample; matching is then performed between the selected subsets.

Each supported test object's weight is distributed among its compatible reference objects in proportion to their prior weights. The accumulated reference contributions are used to reconstruct distributions of properties such as spectroscopic redshift or stellar mass.

Reference PDZs can be integrated when reconstructing redshift distributions. The matching results can be reused for different properties or histogram bins.

Unlike the calibration classes' `get_distribution()` method, `Reconstruction` transfers information from a reference sample to a separate test population.

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

The API docstrings describe class parameters, methods, and usage conventions. The documentation source is included in the `docs/` directory.

To build the HTML documentation locally:

```bash
python -m pip install sphinx furo
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to browse the generated documentation.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use `FOMO-Z` in your research, please cite: