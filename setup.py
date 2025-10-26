from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fomoz",
    version="0.1.0",
    author="Giorgio Francesco Lesci",
    author_email="lesci.giorgio@gmail.com",
    description="Optimized galaxy color selections and photometric redshift estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
        ],
    },
    keywords="astronomy, photometry, galaxy, color-selection, photometric-redshifts",
    url="https://github.com/GiorgioLesci/FOMO-Z",
    project_urls={
        "Bug Reports": "https://github.com/GiorgioLesci/FOMO-Z/issues",
        "Source": "https://github.com/GiorgioLesci/FOMO-Z",
    },
)
