"""Pytest path setup so the documented `pytest` command works from the repo root.

The test suite imports top-level modules (`simulation`, `config`,
`relaxation_gbmc`) and study modules under `studies/` (e.g. `run_n_refinement`).
Adding both directories here removes the need for a manual PYTHONPATH.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "studies")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
