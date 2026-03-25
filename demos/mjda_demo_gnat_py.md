# mjda_demo_gnat_py

This document describes the Python migration of the original `mjda_demo_gnat.m` script.

## Source
- MATLAB: `demos/mjda_demo_gnat.m`
- Python: `demos/mjda_demo_gnat.py`

## Functionality
- Load Gnat repair dataset
- Standardize source and target features
- Run MJDA via `models.mjda.mjda`
- Transform and classify via `classifiers.classifierKNN`
- Evaluate accuracy and F1 via `util.accuracy` and `util.f1score`

## Usage
```bash
python demos/mjda_demo_gnat.py
```
