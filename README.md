# FOMO-Z

A Python package for optimized galaxy color selections and photometric redshift estimation.

## Description

`FOMO-Z` (Flexible Optimizer of Multi-dimensional Operators and photo-Zs) provides tools for creating color-based selections from photometric data and for estimating photometric redshifts of individual galaxies. 

The `binary_classification` subpackage implements two complementary methods for binary classifications, namely `LinearClassification` and `GridClassification`, which optimize the completeness of target samples given a purity threshold, and support a range of galaxy classification and selection tasks.

The `photometric_redshift` subpackage, dedicated to photometric redshift estimation for individual galaxies, will be introduced in a future version of the code.

### LinearClassification

A simple and effective calibration method is implemented in the `LinearClassification` class. It is based on linear color cuts and it is well suited for deriving background galaxy selections.

![DEMO GIF](https://github.com/user-attachments/assets/255bd07f-b66d-40ac-a539-f1429e2cd455)

The figure above shows an example of the optimization process implemented in `LinearClassification`. The assumed limiting redshift is 0.3.

### GridClassification

Complex selections, involving, e.g., requirements on galaxy stellar mass or morphology, may require the use of the `GridClassification` class. Compared to `LinearClassification`, this class implements a more general approach based on discretized color-color grids.

### Performance

The figure below shows the execution time (top) and maximum RAM usage (bottom) of the selection calibration routines implemented in `binary_classification`, as a function of the number of photometric bands used for calibration. The COSMOS20 catalog was used as the reference sample.

![DEMO IMG](https://github.com/user-attachments/assets/a5142986-f632-409a-bc73-a47b51510014)

Blue and red dots show results for `LinearClassification` using 1 and 12 CPUs, respectively. `GridClassification` processing, which is not parallelized, is represented by green crosses. Note that such calibrations are, in principle, performed only once, and then applied to any data via the `ColorSelection` class.

## Installation

### From source
```bash
git clone https://github.com/GiorgioLesci/FOMO-Z.git
cd FOMO-Z
```

To install outside any environment (generally not recommended), run
```bash
pip install .
```

### Conda environment
Alternatively, create the conda environment:
```bash
conda create -n fomoz python=3.9 "numpy<2" scipy astropy pyyaml -c conda-forge
conda activate fomoz
pip install .
```

To activate the environment, run
```bash
conda activate fomoz
```

### Development installation
```bash
pip install -e .[dev]
```

## Documentation

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use `FOMO-Z` in your research, please cite:
