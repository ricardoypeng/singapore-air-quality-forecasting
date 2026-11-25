"""Paths for prepared inputs and notebook model runs."""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PREPARATION = PROJECT / "tmp/_project"
WORKSPACE = next(
    (
        parent for parent in PROJECT.parents
        if parent / "Production-ready_Machine_Learning/Project" == PROJECT.parent
    ),
    PROJECT,
)
DEFAULT_OUTPUT = WORKSPACE / "tmp/_project/pm25-notebook-workflow"
OUTPUT = Path(os.environ.get("PM25_EXPERIMENT_DIR", DEFAULT_OUTPUT)).resolve()
WEATHER = OUTPUT / "causal-weather"
MATRIX = WEATHER / "matrices.npz"
CHECKPOINTS = OUTPUT / "checkpoints-v2"
