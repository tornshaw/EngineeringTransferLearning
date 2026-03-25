# gnat_piper_graph_visualisation_py

This document describes the Python migration of the original `gnat_piper_graph_visualisation.m` script.

## Source
- MATLAB: `demos/gnat_piper_graph_visualisation.m`
- Python: `demos/gnat_piper_graph_visualisation.py`

## Functionality
- Construct Gnat and Piper graphs (with boundary nodes)
- Plot the two graphs side-by-side using `networkx` and `matplotlib`
- Find all boundary-including simple paths with exactly 5 edges

## Usage
```bash
python demos/gnat_piper_graph_visualisation.py
```

## Output
- Graph plots (visual check)
- Number of valid boundary-including path(s)
