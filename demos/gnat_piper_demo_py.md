# gnat_piper_demo_py

This document describes the Python migration of the original `gnat_piper_demo.m` script.

## Source
- MATLAB: `demos/gnat_piper_demo.m`
- Python: `demos/gnat_piper_demo.py`

## Functionality
- Load preprocessed data from `data/gnat_piper_preprocessed_*.mat`
- Normalize source and target features with `StandardScaler`
- Run BDA using `models.bda.bda`
- Compute and display accuracy and F1 (via `util.accuracy` and `util.f1score`)

## Usage
```bash
python demos/gnat_piper_demo.py
```

## Output
- `Zs`, `Zt` (transformed features)
- `Ytp` (pseudo labels)
- `fscore`, `mmd`, `acc`, `f1`
