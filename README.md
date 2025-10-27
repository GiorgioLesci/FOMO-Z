# `FOMO-Z`

A Python package for optimized galaxy color selections and photometric redshift estimation.

## Description

`FOMO-Z` provides tools for creating and optimizing color selections of galaxies using photometric data. It implements methods for finding optimal color cuts that maximize completeness and purity for target redshift ranges.

## Installation

### From source
```bash
git clone https://github.com/GiorgioLesci/FOMO-Z.git
cd FOMO-Z
pip install .
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

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
Run tests and `black` before committing:
```bash
pytest tests/ -v
black fomoz/
```